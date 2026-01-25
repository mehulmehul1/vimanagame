/**
 * User preference schema
 */
interface UserPreferenceSchema {
    // Appearance
    theme: 'dark' | 'light';
    accentColor: string;
    fontSize: 'small' | 'medium' | 'large';

    // Editor behavior
    autoSaveInterval: number; // in minutes
    autoSaveEnabled: boolean;
    createBackupOnSave: boolean;

    // Viewport
    defaultGizmoMode: 'translate' | 'rotate' | 'scale';
    showGrid: boolean;
    showAxes: boolean;
    cameraSmoothness: number;

    // Panels
    panelLayout: 'default' | 'compact' | 'wide';
    hierarchyPanel: { visible: boolean; width: number };
    inspectorPanel: { visible: boolean; width: number };
    timelinePanel: { visible: boolean; height: number };
    consolePanel: { visible: boolean; height: number };

    // Performance
    qualityLevel: 'low' | 'medium' | 'high';
    pixelRatio: number;
    enableFrustumCulling: boolean;
    enableLOD: boolean;

    // Keyboard
    customShortcuts: Record<string, string[]>;

    // Advanced
    developerMode: boolean;
    debugOverlay: boolean;
    verboseLogging: boolean;
}

/**
 * Default preferences
 */
const DEFAULT_PREFERENCES: UserPreferenceSchema = {
    theme: 'dark',
    accentColor: '#4a9eff',
    fontSize: 'medium',

    autoSaveInterval: 2,
    autoSaveEnabled: true,
    createBackupOnSave: true,

    defaultGizmoMode: 'translate',
    showGrid: true,
    showAxes: true,
    cameraSmoothness: 0.5,

    panelLayout: 'default',
    hierarchyPanel: { visible: true, width: 250 },
    inspectorPanel: { visible: true, width: 300 },
    timelinePanel: { visible: false, height: 200 },
    consolePanel: { visible: false, height: 200 },

    qualityLevel: 'medium',
    pixelRatio: 1,
    enableFrustumCulling: true,
    enableLOD: true,

    customShortcuts: {},

    developerMode: false,
    debugOverlay: false,
    verboseLogging: false
};

/**
 * UserPreferences - Manages user settings with localStorage persistence
 *
 * Features:
 * - Settings storage in localStorage
 * - Preferences UI panel (Edit → Preferences)
 * - Options: theme (dark/light), panel layout, auto-save interval, default gizmo mode
 * - Import/export preferences
 *
 * Usage:
 * const prefs = UserPreferences.getInstance();
 * const theme = prefs.get('theme');
 * prefs.set('theme', 'light');
 */
class UserPreferences {
    private static instance: UserPreferences;

    // Preferences storage
    private preferences: UserPreferenceSchema;
    private storageKey: string = 'shadow-web-editor-preferences';

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        // Load preferences from localStorage or use defaults
        this.preferences = this.loadPreferences();
        console.log('UserPreferences: Constructor complete');
    }

    public static getInstance(): UserPreferences {
        if (!UserPreferences.instance) {
            UserPreferences.instance = new UserPreferences();
        }
        return UserPreferences.instance;
    }

    /**
     * Initialize the user preferences manager
     */
    public async initialize(): Promise<void> {
        console.log('UserPreferences: Initializing...');

        // Apply theme
        this.applyTheme(this.preferences.theme);

        // Apply font size
        this.applyFontSize(this.preferences.fontSize);

        this.emit('initialized');
    }

    /**
     * Load preferences from localStorage
     */
    private loadPreferences(): UserPreferenceSchema {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Merge with defaults to handle new properties
                return { ...DEFAULT_PREFERENCES, ...parsed };
            }
        } catch (error) {
            console.error('UserPreferences: Failed to load preferences:', error);
        }
        return { ...DEFAULT_PREFERENCES };
    }

    /**
     * Save preferences to localStorage
     */
    private savePreferences(): void {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.preferences));
            this.emit('saved', this.preferences);
        } catch (error) {
            console.error('UserPreferences: Failed to save preferences:', error);
        }
    }

    /**
     * Get a preference value
     */
    public get<K extends keyof UserPreferenceSchema>(key: K): UserPreferenceSchema[K] {
        return this.preferences[key];
    }

    /**
     * Set a preference value
     */
    public set<K extends keyof UserPreferenceSchema>(key: K, value: UserPreferenceSchema[K]): void {
        const oldValue = this.preferences[key];
        this.preferences[key] = value;

        // Trigger side effects for certain preferences
        if (key === 'theme') {
            this.applyTheme(value as 'dark' | 'light');
        } else if (key === 'fontSize') {
            this.applyFontSize(value as 'small' | 'medium' | 'large');
        }

        this.savePreferences();
        this.emit('changed', { key, value, oldValue });
    }

    /**
     * Get all preferences
     */
    public getAll(): Readonly<UserPreferenceSchema> {
        return { ...this.preferences };
    }

    /**
     * Set multiple preferences at once
     */
    public setMany(updates: Partial<UserPreferenceSchema>): void {
        const changes: Array<{ key: string; value: any; oldValue: any }> = [];

        for (const [key, value] of Object.entries(updates)) {
            const oldValue = this.preferences[key as keyof UserPreferenceSchema];
            this.preferences[key as keyof UserPreferenceSchema] = value as any;
            changes.push({ key, value, oldValue });
        }

        // Apply side effects
        if ('theme' in updates) {
            this.applyTheme(updates.theme as 'dark' | 'light');
        }
        if ('fontSize' in updates) {
            this.applyFontSize(updates.fontSize as 'small' | 'medium' | 'large');
        }

        this.savePreferences();
        this.emit('changedMany', changes);
    }

    /**
     * Apply theme to document
     */
    private applyTheme(theme: 'dark' | 'light'): void {
        if (theme === 'dark') {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
        } else {
            document.body.classList.add('light-theme');
            document.body.classList.remove('dark-theme');
        }
    }

    /**
     * Apply font size to document
     */
    private applyFontSize(size: 'small' | 'medium' | 'large'): void {
        document.body.classList.remove('font-small', 'font-medium', 'font-large');
        document.body.classList.add(`font-${size}`);
    }

    /**
     * Reset to default preferences
     */
    public resetToDefaults(): void {
        const oldPreferences = { ...this.preferences };
        this.preferences = { ...DEFAULT_PREFERENCES };

        // Apply defaults
        this.applyTheme(this.preferences.theme);
        this.applyFontSize(this.preferences.fontSize);

        this.savePreferences();
        this.emit('reset', { oldPreferences, newPreferences: this.preferences });
    }

    /**
     * Export preferences to JSON string
     */
    public exportToString(): string {
        return JSON.stringify(this.preferences, null, 2);
    }

    /**
     * Export preferences as downloadable file
     */
    public exportToFile(): void {
        const dataStr = JSON.stringify(this.preferences, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = 'shadow-web-editor-preferences.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);
        this.emit('exported', this.preferences);
    }

    /**
     * Import preferences from JSON string
     */
    public importFromString(jsonString: string): boolean {
        try {
            const imported = JSON.parse(jsonString);

            // Validate basic structure
            if (typeof imported !== 'object' || imported === null) {
                throw new Error('Invalid preferences format');
            }

            // Merge with defaults to ensure all properties exist
            this.preferences = { ...DEFAULT_PREFERENCES, ...imported };

            // Apply side effects
            this.applyTheme(this.preferences.theme);
            this.applyFontSize(this.preferences.fontSize);

            this.savePreferences();
            this.emit('imported', this.preferences);
            return true;
        } catch (error) {
            console.error('UserPreferences: Failed to import preferences:', error);
            return false;
        }
    }

    /**
     * Import preferences from file
     */
    public importFromFile(file: File): Promise<boolean> {
        return new Promise((resolve) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                const content = event.target?.result as string;
                resolve(this.importFromString(content));
            };

            reader.onerror = () => {
                console.error('UserPreferences: Failed to read file');
                resolve(false);
            };

            reader.readAsText(file);
        });
    }

    /**
     * Get available themes
     */
    public static getAvailableThemes(): Array<{ value: 'dark' | 'light'; label: string }> {
        return [
            { value: 'dark', label: 'Dark' },
            { value: 'light', label: 'Light' }
        ];
    }

    /**
     * Get available font sizes
     */
    public static getAvailableFontSizes(): Array<{ value: 'small' | 'medium' | 'large'; label: string }> {
        return [
            { value: 'small', label: 'Small (12px)' },
            { value: 'medium', label: 'Medium (14px)' },
            { value: 'large', label: 'Large (16px)' }
        ];
    }

    /**
     * Get available quality levels
     */
    public static getAvailableQualityLevels(): Array<{ value: 'low' | 'medium' | 'high'; label: string }> {
        return [
            { value: 'low', label: 'Low (0.5x)' },
            { value: 'medium', label: 'Medium (1x)' },
            { value: 'high', label: 'High (2x)' }
        ];
    }

    /**
     * Get available gizmo modes
     */
    public static getAvailableGizmoModes(): Array<{ value: 'translate' | 'rotate' | 'scale'; label: string }> {
        return [
            { value: 'translate', label: 'Translate (G)' },
            { value: 'rotate', label: 'Rotate (R)' },
            { value: 'scale', label: 'Scale (S)' }
        ];
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
                    console.error(`UserPreferences: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }
}

export default UserPreferences;
