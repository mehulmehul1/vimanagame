import * as THREE from 'three';

/**
 * Command interface for undo/redo operations
 */
export interface Command {
    execute(): void;
    undo(): void;
    canUndo(): boolean;
    getDescription(): string;
}

/**
 * Transform Command - For position, rotation, scale changes
 */
class TransformCommand implements Command {
    private object: THREE.Object3D;
    private oldPosition: THREE.Vector3;
    private oldRotation: THREE.Euler;
    private oldScale: THREE.Vector3;
    private newPosition: THREE.Vector3;
    private newRotation: THREE.Euler;
    private newScale: THREE.Vector3;

    constructor(
        object: THREE.Object3D,
        newPosition: THREE.Vector3,
        newRotation: THREE.Euler,
        newScale: THREE.Vector3
    ) {
        this.object = object;
        this.oldPosition = object.position.clone();
        this.oldRotation = object.rotation.clone();
        this.oldScale = object.scale.clone();
        this.newPosition = newPosition.clone();
        this.newRotation = newRotation.clone();
        this.newScale = newScale.clone();
    }

    execute(): void {
        this.object.position.copy(this.newPosition);
        this.object.rotation.copy(this.newRotation);
        this.object.scale.copy(this.newScale);
    }

    undo(): void {
        this.object.position.copy(this.oldPosition);
        this.object.rotation.copy(this.oldRotation);
        this.object.scale.copy(this.oldScale);
    }

    canUndo(): boolean {
        return true;
    }

    getDescription(): string {
        return `Transform ${this.object.name || 'Object'}`;
    }
}

/**
 * Reparent Command - For changing object parent
 */
class ReparentCommand implements Command {
    private object: THREE.Object3D;
    private oldParent: THREE.Object3D | null;
    private newParent: THREE.Object3D | null;
    private scene: THREE.Scene;

    constructor(object: THREE.Object3D, newParent: THREE.Object3D | null, scene: THREE.Scene) {
        this.object = object;
        this.oldParent = object.parent;
        this.newParent = newParent;
        this.scene = scene;
    }

    execute(): void {
        if (this.newParent) {
            this.newParent.add(this.object);
        } else {
            this.scene.add(this.object);
        }
    }

    undo(): void {
        if (this.oldParent) {
            this.oldParent.add(this.object);
        } else {
            this.scene.add(this.object);
        }
    }

    canUndo(): boolean {
        return true;
    }

    getDescription(): string {
        return `Reparent ${this.object.name || 'Object'}`;
    }
}

/**
 * Add Object Command
 */
class AddObjectCommand implements Command {
    private object: THREE.Object3D;
    private parent: THREE.Object3D;
    private added: boolean = false;

    constructor(object: THREE.Object3D, parent: THREE.Object3D) {
        this.object = object;
        this.parent = parent;
    }

    execute(): void {
        if (!this.added) {
            this.parent.add(this.object);
            this.added = true;
        }
    }

    undo(): void {
        this.parent.remove(this.object);
        this.added = false;
    }

    canUndo(): boolean {
        return this.added;
    }

    getDescription(): string {
        return `Add ${this.object.name || 'Object'}`;
    }
}

/**
 * Delete Object Command
 */
class DeleteObjectCommand implements Command {
    private object: THREE.Object3D;
    private parent: THREE.Object3D | null;
    private scene: THREE.Scene;
    private deleted: boolean = false;

    constructor(object: THREE.Object3D, scene: THREE.Scene) {
        this.object = object;
        this.parent = object.parent;
        this.scene = scene;
    }

    execute(): void {
        if (!this.deleted) {
            if (this.parent) {
                this.parent.remove(this.object);
            } else {
                this.scene.remove(this.object);
            }
            this.deleted = true;
        }
    }

    undo(): void {
        if (this.deleted) {
            if (this.parent) {
                this.parent.add(this.object);
            } else {
                this.scene.add(this.object);
            }
            this.deleted = false;
        }
    }

    canUndo(): boolean {
        return this.deleted;
    }

    getDescription(): string {
        return `Delete ${this.object.name || 'Object'}`;
    }
}

/**
 * Property Change Command - For any property change
 */
class PropertyChangeCommand implements Command {
    private object: any;
    private property: string;
    private oldValue: any;
    private newValue: any;

    constructor(object: any, property: string, newValue: any) {
        this.object = object;
        this.property = property;
        this.oldValue = object[property];
        this.newValue = newValue;
    }

    execute(): void {
        this.object[this.property] = this.newValue;
    }

    undo(): void {
        this.object[this.property] = this.oldValue;
    }

    canUndo(): boolean {
        return true;
    }

    getDescription(): string {
        return `Change ${this.property} of ${this.object.name || 'Object'}`;
    }
}

/**
 * UndoRedoManager - Command pattern implementation for undo/redo
 *
 * Features:
 * - Command history stack (max 100)
 * - Undo (Ctrl+Z)
 * - Redo (Ctrl+Shift+Z or Ctrl+Y)
 * - Commands for: Transform, Reparent, Add, Delete, Property Change
 * - Event emission for UI updates
 */
class UndoRedoManager {
    private static instance: UndoRedoManager;

    private undoStack: Command[] = [];
    private redoStack: Command[] = [];
    private maxHistory: number = 100;

    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        console.log('UndoRedoManager: Initialized');
        this.setupKeyboardShortcuts();
    }

    public static getInstance(): UndoRedoManager {
        if (!UndoRedoManager.instance) {
            UndoRedoManager.instance = new UndoRedoManager();
        }
        return UndoRedoManager.instance;
    }

    private setupKeyboardShortcuts(): void {
        const handleKeyDown = (event: KeyboardEvent) => {
            // Ignore if typing in input
            if (event.target instanceof HTMLInputElement ||
                event.target instanceof HTMLTextAreaElement) {
                return;
            }

            // Ctrl+Z for undo
            if ((event.ctrlKey || event.metaKey) && event.key === 'z' && !event.shiftKey) {
                event.preventDefault();
                this.undo();
                return;
            }

            // Ctrl+Shift+Z or Ctrl+Y for redo
            if ((event.ctrlKey || event.metaKey) && (event.key === 'y' || (event.key === 'z' && event.shiftKey))) {
                event.preventDefault();
                this.redo();
                return;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
    }

    /**
     * Execute a command and add to undo stack
     */
    public executeCommand(command: Command): void {
        command.execute();
        this.undoStack.push(command);

        // Limit stack size
        if (this.undoStack.length > this.maxHistory) {
            this.undoStack.shift();
        }

        // Clear redo stack when new command is executed
        this.redoStack = [];

        console.log('UndoRedoManager: Executed command:', command.getDescription());
        this.emit('commandExecuted', { command, description: command.getDescription() });
        this.emit('historyChanged');
    }

    /**
     * Undo last command
     */
    public undo(): void {
        const command = this.undoStack.pop();
        if (!command) {
            console.warn('UndoRedoManager: Nothing to undo');
            return;
        }

        if (!command.canUndo()) {
            console.warn('UndoRedoManager: Command cannot be undone');
            return;
        }

        command.undo();
        this.redoStack.push(command);

        console.log('UndoRedoManager: Undone command:', command.getDescription());
        this.emit('commandUndone', { command, description: command.getDescription() });
        this.emit('historyChanged');
    }

    /**
     * Redo last undone command
     */
    public redo(): void {
        const command = this.redoStack.pop();
        if (!command) {
            console.warn('UndoRedoManager: Nothing to redo');
            return;
        }

        command.execute();
        this.undoStack.push(command);

        console.log('UndoRedoManager: Redone command:', command.getDescription());
        this.emit('commandRedone', { command, description: command.getDescription() });
        this.emit('historyChanged');
    }

    /**
     * Check if can undo
     */
    public canUndo(): boolean {
        return this.undoStack.length > 0;
    }

    /**
     * Check if can redo
     */
    public canRedo(): boolean {
        return this.redoStack.length > 0;
    }

    /**
     * Get undo stack size
     */
    public getUndoCount(): number {
        return this.undoStack.length;
    }

    /**
     * Get redo stack size
     */
    public getRedoCount(): number {
        return this.redoStack.length;
    }

    /**
     * Clear all history
     */
    public clearHistory(): void {
        this.undoStack = [];
        this.redoStack = [];
        console.log('UndoRedoManager: History cleared');
        this.emit('historyChanged');
    }

    /**
     * Create and execute a transform command
     */
    public executeTransform(
        object: THREE.Object3D,
        newPosition: THREE.Vector3,
        newRotation: THREE.Euler,
        newScale: THREE.Vector3
    ): void {
        const command = new TransformCommand(object, newPosition, newRotation, newScale);
        this.executeCommand(command);
    }

    /**
     * Create and execute a reparent command
     */
    public executeReparent(object: THREE.Object3D, newParent: THREE.Object3D | null, scene: THREE.Scene): void {
        const command = new ReparentCommand(object, newParent, scene);
        this.executeCommand(command);
    }

    /**
     * Create and execute an add object command
     */
    public executeAdd(object: THREE.Object3D, parent: THREE.Object3D): void {
        const command = new AddObjectCommand(object, parent);
        this.executeCommand(command);
    }

    /**
     * Create and execute a delete object command
     */
    public executeDelete(object: THREE.Object3D, scene: THREE.Scene): void {
        const command = new DeleteObjectCommand(object, scene);
        this.executeCommand(command);
    }

    /**
     * Create and execute a property change command
     */
    public executePropertyChange(object: any, property: string, newValue: any): void {
        const command = new PropertyChangeCommand(object, property, newValue);
        this.executeCommand(command);
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
                    console.error(`UndoRedoManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Destroy and clean up
     */
    public destroy(): void {
        this.clearHistory();
        this.eventListeners.clear();
        console.log('UndoRedoManager: Destroyed');
    }
}

export default UndoRedoManager;
export { TransformCommand, ReparentCommand, AddObjectCommand, DeleteObjectCommand, PropertyChangeCommand };
