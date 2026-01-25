/**
 * FileWatcher - Watch files for changes and emit events
 *
 * This class uses the File System Access API or falls back to polling
 * to detect file changes. Since we're in a browser environment,
 * we'll use polling via fetch with cache-busting.
 *
 * Features:
 * - Watch sceneData.js for changes
 * - Debounced change detection (500ms default)
 * - Auto-reload capability
 * - Event emission for UI updates
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */
class FileWatcher {
    // Singleton instance
    private static instance: FileWatcher;

    // Watched files
    private watchedFiles: Map<string, {
        lastModified: number;
        intervalId: number;
        debounceTimer: number | null;
        callbacks: Set<(file: string) => void>;
    }> = new Map();

    // Polling interval (default: 1000ms)
    private pollInterval: number = 1000;

    // Debounce delay (default: 500ms)
    private debounceDelay: number = 500;

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        console.log('FileWatcher: Constructor complete');
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): FileWatcher {
        if (!FileWatcher.instance) {
            FileWatcher.instance = new FileWatcher();
        }
        return FileWatcher.instance;
    }

    /**
     * Watch a file for changes
     *
     * @param filePath - Path to the file to watch
     * @param callback - Callback function when file changes
     * @param pollInterval - Optional custom poll interval (default: 1000ms)
     */
    public async watch(filePath: string, callback: (file: string) => void, pollInterval?: number): Promise<void> {
        console.log('FileWatcher: Watching file', filePath);

        // Stop watching if already watching
        if (this.watchedFiles.has(filePath)) {
            this.unwatch(filePath);
        }

        try {
            // Get initial file timestamp
            const lastModified = await this.getFileTimestamp(filePath);

            // Setup file watcher
            const interval = pollInterval || this.pollInterval;

            const intervalId = window.setInterval(async () => {
                try {
                    const newTimestamp = await this.getFileTimestamp(filePath);

                    const watchedFile = this.watchedFiles.get(filePath);
                    if (!watchedFile) return;

                    // Check if file has been modified
                    if (newTimestamp > watchedFile.lastModified) {
                        console.log('FileWatcher: File modified', filePath);

                        // Debounce the change notification
                        if (watchedFile.debounceTimer) {
                            clearTimeout(watchedFile.debounceTimer);
                        }

                        watchedFile.debounceTimer = window.setTimeout(() => {
                            watchedFile.lastModified = newTimestamp;

                            // Notify all callbacks
                            watchedFile.callbacks.forEach(cb => {
                                try {
                                    cb(filePath);
                                } catch (error) {
                                    console.error('FileWatcher: Error in callback:', error);
                                }
                            });

                            // Emit global event
                            this.emit('fileChanged', { file: filePath, timestamp: newTimestamp });

                            watchedFile.debounceTimer = null;
                        }, this.debounceDelay);
                    }
                } catch (error) {
                    console.error('FileWatcher: Error checking file:', error);
                }
            }, interval);

            // Store watcher
            this.watchedFiles.set(filePath, {
                lastModified,
                intervalId,
                debounceTimer: null,
                callbacks: new Set([callback])
            });

            this.emit('watchStarted', { file: filePath });
        } catch (error) {
            console.error('FileWatcher: Failed to start watching file:', error);
            throw error;
        }
    }

    /**
     * Stop watching a file
     */
    public unwatch(filePath: string): void {
        const watchedFile = this.watchedFiles.get(filePath);
        if (!watchedFile) {
            console.warn('FileWatcher: File not being watched', filePath);
            return;
        }

        // Clear interval
        clearInterval(watchedFile.intervalId);

        // Clear debounce timer
        if (watchedFile.debounceTimer) {
            clearTimeout(watchedFile.debounceTimer);
        }

        // Remove from watched files
        this.watchedFiles.delete(filePath);

        console.log('FileWatcher: Stopped watching', filePath);
        this.emit('watchStopped', { file: filePath });
    }

    /**
     * Get file timestamp via fetch with cache-busting
     */
    private async getFileTimestamp(filePath: string): Promise<number> {
        try {
            // Add cache-busting parameter
            const cacheBuster = `?_t=${Date.now()}`;
            const response = await fetch(filePath + cacheBuster, { method: 'HEAD' });

            if (!response.ok) {
                throw new Error(`Failed to fetch file: ${response.status} ${response.statusText}`);
            }

            // Get last-modified header
            const lastModified = response.headers.get('last-modified');
            if (lastModified) {
                return new Date(lastModified).getTime();
            }

            // Fallback: return current time if no last-modified header
            return Date.now();
        } catch (error) {
            console.error('FileWatcher: Failed to get file timestamp:', error);
            // Return current time to avoid constant false positives
            return Date.now();
        }
    }

    /**
     * Add callback to existing watcher
     */
    public addCallback(filePath: string, callback: (file: string) => void): void {
        const watchedFile = this.watchedFiles.get(filePath);
        if (!watchedFile) {
            console.warn('FileWatcher: File not being watched', filePath);
            return;
        }

        watchedFile.callbacks.add(callback);
        console.log('FileWatcher: Added callback for', filePath);
    }

    /**
     * Remove callback from watcher
     */
    public removeCallback(filePath: string, callback: (file: string) => void): void {
        const watchedFile = this.watchedFiles.get(filePath);
        if (!watchedFile) {
            console.warn('FileWatcher: File not being watched', filePath);
            return;
        }

        watchedFile.callbacks.delete(callback);
        console.log('FileWatcher: Removed callback for', filePath);
    }

    /**
     * Set debounce delay
     */
    public setDebounceDelay(delay: number): void {
        this.debounceDelay = delay;
        console.log('FileWatcher: Debounce delay set to', delay, 'ms');
    }

    /**
     * Set polling interval
     */
    public setPollInterval(interval: number): void {
        this.pollInterval = interval;
        console.log('FileWatcher: Poll interval set to', interval, 'ms');
    }

    /**
     * Stop all watchers
     */
    public stopAll(): void {
        const files = Array.from(this.watchedFiles.keys());
        files.forEach(file => this.unwatch(file));
        console.log('FileWatcher: Stopped all watchers');
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
                    console.error(`FileWatcher: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Destroy and clean up
     */
    public destroy(): void {
        this.stopAll();
        this.eventListeners.clear();
        console.log('FileWatcher: Destroyed');
    }
}

export default FileWatcher;
