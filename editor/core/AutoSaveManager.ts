import EditorManager from './EditorManager.js';
import DataManager from './DataManager.js';

/**
 * AutoSaveManager - Automatic scene saving system
 *
 * Features:
 * - Configurable auto-save interval (default: 2 minutes)
 * - Save before play mode
 * - Save on file change
 * - Recovery from crash (localStorage backup)
 * - Auto-save indicator
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */
class AutoSaveManager {
    // Singleton instance
    private static instance: AutoSaveManager;

    // Auto-save interval (default: 2 minutes)
    private autoSaveInterval: number = 2 * 60 * 1000; // 2 minutes in ms

    // Interval timer
    private intervalTimer: number | null = null;

    // Auto-save enabled flag
    private autoSaveEnabled: boolean = false;

    // Last save timestamp
    private lastSaveTime: number = 0;

    // LocalStorage key for crash recovery
    private STORAGE_KEY = 'shadow-editor-autosave-backup';

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    // Managers
    private editorManager: EditorManager;
    private dataManager: DataManager;

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        this.editorManager = EditorManager.getInstance();
        this.dataManager = DataManager.getInstance();
        console.log('AutoSaveManager: Constructor complete');
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): AutoSaveManager {
        if (!AutoSaveManager.instance) {
            AutoSaveManager.instance = new AutoSaveManager();
        }
        return AutoSaveManager.instance;
    }

    /**
     * Enable auto-save
     *
     * @param intervalMinutes - Auto-save interval in minutes (default: 2)
     */
    public enable(intervalMinutes: number = 2): void {
        if (this.autoSaveEnabled) {
            console.warn('AutoSaveManager: Auto-save already enabled');
            return;
        }

        this.autoSaveInterval = intervalMinutes * 60 * 1000;
        this.autoSaveEnabled = true;

        console.log('AutoSaveManager: Enabled with interval', intervalMinutes, 'minutes');

        // Start auto-save timer
        this.startAutoSaveTimer();

        // Listen for play mode changes to save before play
        this.editorManager.on('playModeEntered', () => {
            this.saveNow('before play mode');
        });

        // Listen for file changes
        this.dataManager.on('dataUpdated', () => {
            this.saveToLocalStorage();
        });

        // Check for crash recovery
        this.checkCrashRecovery();

        this.emit('autoSaveEnabled', { interval: this.autoSaveInterval });
    }

    /**
     * Disable auto-save
     */
    public disable(): void {
        if (!this.autoSaveEnabled) {
            console.warn('AutoSaveManager: Auto-save already disabled');
            return;
        }

        this.autoSaveEnabled = false;

        // Stop timer
        if (this.intervalTimer !== null) {
            clearInterval(this.intervalTimer);
            this.intervalTimer = null;
        }

        console.log('AutoSaveManager: Disabled');
        this.emit('autoSaveDisabled', {});
    }

    /**
     * Start auto-save timer
     */
    private startAutoSaveTimer(): void {
        if (this.intervalTimer !== null) {
            clearInterval(this.intervalTimer);
        }

        this.intervalTimer = window.setInterval(() => {
            this.saveNow('scheduled');
        }, this.autoSaveInterval);

        console.log('AutoSaveManager: Timer started (interval:', this.autoSaveInterval, 'ms)');
    }

    /**
     * Save immediately
     */
    public async saveNow(reason: string = 'manual'): Promise<void> {
        if (!this.autoSaveEnabled && reason !== 'manual') {
            return;
        }

        console.log('AutoSaveManager: Saving scene (reason:', reason + ')');

        try {
            // Update scene data from Three.js scene
            await this.dataManager.updateSceneData();

            // Save to localStorage for crash recovery
            this.saveToLocalStorage();

            // Update last save time
            this.lastSaveTime = Date.now();

            // Emit event
            this.emit('autoSaved', {
                timestamp: this.lastSaveTime,
                reason
            });

            console.log('AutoSaveManager: Scene saved successfully');
        } catch (error) {
            console.error('AutoSaveManager: Failed to save scene:', error);
            this.emit('autoSaveError', { error, reason });
        }
    }

    /**
     * Save to localStorage for crash recovery
     */
    private saveToLocalStorage(): void {
        try {
            const sceneData = this.dataManager.getSceneData();
            const backup = {
                timestamp: Date.now(),
                sceneData: sceneData
            };

            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(backup));
            console.log('AutoSaveManager: Backup saved to localStorage');
        } catch (error) {
            console.error('AutoSaveManager: Failed to save to localStorage:', error);
        }
    }

    /**
     * Check for crash recovery
     */
    private checkCrashRecovery(): void {
        try {
            const backupStr = localStorage.getItem(this.STORAGE_KEY);

            if (backupStr) {
                const backup = JSON.parse(backupStr);
                const backupTime = backup.timestamp || 0;
                const now = Date.now();
                const hoursSinceBackup = (now - backupTime) / (1000 * 60 * 60);

                // Only offer recovery if backup is less than 24 hours old
                if (hoursSinceBackup < 24) {
                    console.log('AutoSaveManager: Found crash recovery backup from', new Date(backupTime));
                    this.emit('crashRecoveryAvailable', { backup, backupTime });
                } else {
                    // Backup is too old, remove it
                    localStorage.removeItem(this.STORAGE_KEY);
                }
            }
        } catch (error) {
            console.error('AutoSaveManager: Failed to check crash recovery:', error);
        }
    }

    /**
     * Recover from crash
     */
    public async recoverFromCrash(): Promise<boolean> {
        try {
            const backupStr = localStorage.getItem(this.STORAGE_KEY);

            if (!backupStr) {
                console.warn('AutoSaveManager: No crash recovery backup found');
                return false;
            }

            const backup = JSON.parse(backupStr);

            // Load scene data from backup
            await this.dataManager.buildSceneFromData(backup.sceneData);

            console.log('AutoSaveManager: Scene recovered from crash backup');

            // Clear backup after successful recovery
            localStorage.removeItem(this.STORAGE_KEY);

            this.emit('crashRecovered', { backup });
            return true;
        } catch (error) {
            console.error('AutoSaveManager: Failed to recover from crash:', error);
            return false;
        }
    }

    /**
     * Clear crash recovery backup
     */
    public clearCrashBackup(): void {
        localStorage.removeItem(this.STORAGE_KEY);
        console.log('AutoSaveManager: Crash recovery backup cleared');
    }

    /**
     * Get last save time
     */
    public getLastSaveTime(): number {
        return this.lastSaveTime;
    }

    /**
     * Get time until next save
     */
    public getTimeUntilNextSave(): number {
        if (!this.autoSaveEnabled) {
            return 0;
        }

        const elapsed = Date.now() - this.lastSaveTime;
        const remaining = this.autoSaveInterval - elapsed;

        return Math.max(0, remaining);
    }

    /**
     * Set auto-save interval
     */
    public setAutoSaveInterval(intervalMinutes: number): void {
        this.autoSaveInterval = intervalMinutes * 60 * 1000;

        if (this.autoSaveEnabled) {
            this.startAutoSaveTimer();
        }

        console.log('AutoSaveManager: Interval set to', intervalMinutes, 'minutes');
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
                    console.error(`AutoSaveManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Destroy and clean up
     */
    public destroy(): void {
        this.disable();
        this.eventListeners.clear();
        console.log('AutoSaveManager: Destroyed');
    }
}

export default AutoSaveManager;
