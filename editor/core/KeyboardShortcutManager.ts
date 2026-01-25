/**
 * Keyboard shortcut definition
 */
interface KeyboardShortcut {
    id: string;
    keys: string[]; // Key combinations (e.g., ['Ctrl', 'S'], ['G'])
    description: string;
    context?: 'global' | 'editor' | 'viewport' | 'hierarchy' | 'inspector';
    action: () => void;
    enabled: boolean;
}

/**
 * Shortcut category for reference panel
 */
interface ShortcutCategory {
    name: string;
    shortcuts: KeyboardShortcut[];
}

/**
 * KeyboardShortcutManager - Global keyboard shortcuts system
 *
 * Features:
 * - Global shortcuts: Ctrl+S (save), Ctrl+Z (undo), Ctrl+Shift+Z (redo), etc.
 * - Context-sensitive shortcuts (G/R/S for gizmos when object selected)
 * - Shortcut reference panel (F1)
 * - Configurable via user preferences
 *
 * Usage:
 * KeyboardShortcutManager.getInstance().register({
 *     id: 'my-shortcut',
 *     keys: ['Ctrl', 'K'],
 *     description: 'My action',
 *     action: () => console.log('Action triggered')
 * });
 */
class KeyboardShortcutManager {
    private static instance: KeyboardShortcutManager;

    // Registered shortcuts
    private shortcuts: Map<string, KeyboardShortcut> = new Map();

    // Active context
    private currentContext: KeyboardShortcut['context'] = 'global';

    // State
    private enabled: boolean = true;
    private showReference: boolean = false;

    // Key state for combinations
    private pressedKeys: Set<string> = new Set();

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.setupEventListeners();
        this.registerDefaultShortcuts();
        console.log('KeyboardShortcutManager: Constructor complete');
    }

    public static getInstance(): KeyboardShortcutManager {
        if (!KeyboardShortcutManager.instance) {
            KeyboardShortcutManager.instance = new KeyboardShortcutManager();
        }
        return KeyboardShortcutManager.instance;
    }

    /**
     * Initialize the keyboard shortcut manager
     */
    public async initialize(): Promise<void> {
        console.log('KeyboardShortcutManager: Initializing...');
        this.emit('initialized');
    }

    /**
     * Setup event listeners
     */
    private setupEventListeners(): void {
        window.addEventListener('keydown', this.handleKeyDown);
        window.addEventListener('keyup', this.handleKeyUp);
    }

    /**
     * Register default shortcuts
     */
    private registerDefaultShortcuts(): void {
        // File operations
        this.register({
            id: 'file.save',
            keys: ['Ctrl', 's'],
            description: 'Save scene',
            context: 'global',
            action: () => this.emit('save'),
            enabled: true
        });

        this.register({
            id: 'file.export',
            keys: ['Ctrl', 'Shift', 'E'],
            description: 'Export scene',
            context: 'global',
            action: () => this.emit('export'),
            enabled: true
        });

        // Edit operations
        this.register({
            id: 'edit.undo',
            keys: ['Ctrl', 'z'],
            description: 'Undo',
            context: 'global',
            action: () => this.emit('undo'),
            enabled: true
        });

        this.register({
            id: 'edit.redo',
            keys: ['Ctrl', 'Shift', 'Z'],
            description: 'Redo',
            context: 'global',
            action: () => this.emit('redo'),
            enabled: true
        });

        this.register({
            id: 'edit.duplicate',
            keys: ['Ctrl', 'd'],
            description: 'Duplicate selected',
            context: 'editor',
            action: () => this.emit('duplicate'),
            enabled: true
        });

        this.register({
            id: 'edit.delete',
            keys: ['Delete'],
            description: 'Delete selected',
            context: 'editor',
            action: () => this.emit('delete'),
            enabled: true
        });

        this.register({
            id: 'edit.rename',
            keys: ['F2'],
            description: 'Rename selected',
            context: 'hierarchy',
            action: () => this.emit('rename'),
            enabled: true
        });

        // Gizmo modes
        this.register({
            id: 'gizmo.translate',
            keys: ['G'],
            description: 'Translate mode',
            context: 'viewport',
            action: () => this.emit('gizmoMode', 'translate'),
            enabled: true
        });

        this.register({
            id: 'gizmo.rotate',
            keys: ['R'],
            description: 'Rotate mode',
            context: 'viewport',
            action: () => this.emit('gizmoMode', 'rotate'),
            enabled: true
        });

        this.register({
            id: 'gizmo.scale',
            keys: ['S'],
            description: 'Scale mode',
            context: 'viewport',
            action: () => this.emit('gizmoMode', 'scale'),
            enabled: true
        });

        // View operations
        this.register({
            id: 'view.focus',
            keys: ['F'],
            description: 'Focus on selection',
            context: 'viewport',
            action: () => this.emit('focusSelection'),
            enabled: true
        });

        this.register({
            id: 'view.frame',
            keys: ['Shift', 'F'],
            description: 'Frame selection',
            context: 'viewport',
            action: () => this.emit('frameSelection'),
            enabled: true
        });

        // Panel toggles
        this.register({
            id: 'panel.hierarchy',
            keys: ['Ctrl', '1'],
            description: 'Toggle Hierarchy',
            context: 'global',
            action: () => this.emit('togglePanel', 'hierarchy'),
            enabled: true
        });

        this.register({
            id: 'panel.inspector',
            keys: ['Ctrl', '2'],
            description: 'Toggle Inspector',
            context: 'global',
            action: () => this.emit('togglePanel', 'inspector'),
            enabled: true
        });

        this.register({
            id: 'panel.timeline',
            keys: ['Ctrl', '3'],
            description: 'Toggle Timeline',
            context: 'global',
            action: () => this.emit('togglePanel', 'timeline'),
            enabled: true
        });

        this.register({
            id: 'panel.console',
            keys: ['Ctrl', '4'],
            description: 'Toggle Console',
            context: 'global',
            action: () => this.emit('togglePanel', 'console'),
            enabled: true
        });

        // Play mode
        this.register({
            id: 'play.toggle',
            keys: ['Ctrl', 'p'],
            description: 'Toggle play mode',
            context: 'global',
            action: () => this.emit('togglePlayMode'),
            enabled: true
        });

        // Help
        this.register({
            id: 'help.shortcuts',
            keys: ['F1'],
            description: 'Show keyboard shortcuts',
            context: 'global',
            action: () => {
                this.showReference = !this.showReference;
                this.emit('showShortcuts', this.showReference);
            },
            enabled: true
        });

        // Debug toggles
        this.register({
            id: 'debug.stats',
            keys: ['Ctrl', 'Shift', 'S'],
            description: 'Toggle stats',
            context: 'global',
            action: () => this.emit('toggleStats'),
            enabled: true
        });

        console.log('KeyboardShortcutManager: Registered default shortcuts');
    }

    /**
     * Register a keyboard shortcut
     */
    public register(shortcut: KeyboardShortcut): void {
        this.shortcuts.set(shortcut.id, shortcut);
        console.log(`KeyboardShortcutManager: Registered shortcut "${shortcut.id}"`);
    }

    /**
     * Unregister a keyboard shortcut
     */
    public unregister(id: string): void {
        this.shortcuts.delete(id);
        console.log(`KeyboardShortcutManager: Unregistered shortcut "${id}"`);
    }

    /**
     * Handle key down event
     */
    private handleKeyDown = (event: KeyboardEvent): void => {
        if (!this.enabled) return;

        // Add key to pressed keys set
        this.pressedKeys.add(event.key.toLowerCase());
        this.pressedKeys.add(event.code);

        // Check for matching shortcuts
        const matchingShortcut = this.findMatchingShortcut();

        if (matchingShortcut) {
            event.preventDefault();
            event.stopPropagation();

            // Execute action
            if (matchingShortcut.enabled) {
                matchingShortcut.action();
                this.emit('shortcutTriggered', {
                    id: matchingShortcut.id,
                    keys: matchingShortcut.keys
                });
            }
        }
    };

    /**
     * Handle key up event
     */
    private handleKeyUp = (event: KeyboardEvent): void => {
        this.pressedKeys.delete(event.key.toLowerCase());
        this.pressedKeys.delete(event.code);
    };

    /**
     * Find a matching shortcut based on pressed keys
     */
    private findMatchingShortcut(): KeyboardShortcut | null {
        for (const shortcut of this.shortcuts.values()) {
            if (!shortcut.enabled) continue;

            // Check context
            if (shortcut.context && shortcut.context !== 'global' && shortcut.context !== this.currentContext) {
                continue;
            }

            // Check if all required keys are pressed
            const requiredKeys = shortcut.keys.map(k => k.toLowerCase());
            const modifiers = requiredKeys.filter(k => ['ctrl', 'shift', 'alt', 'meta'].includes(k));

            // Check modifiers
            if (modifiers.includes('ctrl') && !this.pressedKeys.has('control')) continue;
            if (modifiers.includes('shift') && !this.pressedKeys.has('shift')) continue;
            if (modifiers.includes('alt') && !this.pressedKeys.has('alt')) continue;
            if (modifiers.includes('meta') && !this.pressedKeys.has('meta')) continue;

            // Check main key
            const mainKey = requiredKeys.find(k => !['ctrl', 'shift', 'alt', 'meta'].includes(k));
            if (mainKey && this.pressedKeys.has(mainKey)) {
                return shortcut;
            }
        }

        return null;
    }

    /**
     * Set current context
     */
    public setContext(context: KeyboardShortcut['context']): void {
        this.currentContext = context;
    }

    /**
     * Get current context
     */
    public getContext(): KeyboardShortcut['context'] {
        return this.currentContext;
    }

    /**
     * Enable/disable all shortcuts
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        console.log(`KeyboardShortcutManager: ${enabled ? 'Enabled' : 'Disabled'}`);
    }

    /**
     * Enable/disable specific shortcut
     */
    public setShortcutEnabled(id: string, enabled: boolean): void {
        const shortcut = this.shortcuts.get(id);
        if (shortcut) {
            shortcut.enabled = enabled;
        }
    }

    /**
     * Get all shortcuts organized by category
     */
    public getShortcutsByCategory(): ShortcutCategory[] {
        const categories: Map<string, KeyboardShortcut[]> = new Map();

        for (const shortcut of this.shortcuts.values()) {
            // Determine category based on shortcut ID
            let category = 'Other';
            if (shortcut.id.startsWith('file.')) category = 'File';
            else if (shortcut.id.startsWith('edit.')) category = 'Edit';
            else if (shortcut.id.startsWith('gizmo.')) category = 'Transform Gizmos';
            else if (shortcut.id.startsWith('view.')) category = 'View';
            else if (shortcut.id.startsWith('panel.')) category = 'Panels';
            else if (shortcut.id.startsWith('play.')) category = 'Play';
            else if (shortcut.id.startsWith('help.') || shortcut.id.startsWith('debug.')) category = 'Help';

            if (!categories.has(category)) {
                categories.set(category, []);
            }
            categories.get(category)!.push(shortcut);
        }

        return Array.from(categories.entries()).map(([name, shortcuts]) => ({
            name,
            shortcuts
        }));
    }

    /**
     * Format shortcut keys for display
     */
    public formatKeys(keys: string[]): string {
        return keys
            .map(k => {
                if (k === 'Ctrl') return '⌃';
                if (k === 'Shift') return '⇧';
                if (k === 'Alt') return '⌥';
                if (k === 'Meta') return '⌘';
                if (k.length === 1) return k.toUpperCase();
                return k;
            })
            .join(' + ');
    }

    /**
     * Export shortcuts to JSON
     */
    public exportShortcuts(): Record<string, string[]> {
        const exported: Record<string, string[]> = {};
        for (const [id, shortcut] of this.shortcuts.entries()) {
            exported[id] = shortcut.keys;
        }
        return exported;
    }

    /**
     * Import shortcuts from JSON
     */
    public importShortcuts(shortcuts: Record<string, string[]>): void {
        for (const [id, keys] of Object.entries(shortcuts)) {
            const shortcut = this.shortcuts.get(id);
            if (shortcut) {
                shortcut.keys = keys;
            }
        }
        console.log('KeyboardShortcutManager: Imported shortcuts');
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
                    console.error(`KeyboardShortcutManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Clean up
     */
    public destroy(): void {
        window.removeEventListener('keydown', this.handleKeyDown);
        window.removeEventListener('keyup', this.handleKeyUp);
        this.shortcuts.clear();
        this.pressedKeys.clear();
        this.eventListeners.clear();
        console.log('KeyboardShortcutManager: Destroyed');
    }
}

export default KeyboardShortcutManager;
