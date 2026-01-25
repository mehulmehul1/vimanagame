/**
 * ShellManager - Manages shell collectibles across all chambers
 *
 * Singleton pattern for tracking shell collection state globally.
 * Integrates with UI overlay and handles save state persistence.
 */

import * as THREE from 'three';
import { ShellCollectible } from './ShellCollectible';

export interface ShellCollectionState {
    archiveOfVoices: boolean;
    galleryOfForms: boolean;
    hydroponicMemory: boolean;
    engineOfGrowth: boolean;
}

export type ChamberId = keyof ShellCollectionState;

export const CHAMBER_NAMES: Record<ChamberId, string> = {
    archiveOfVoices: 'Archive of Voices',
    galleryOfForms: 'Gallery of Forms',
    hydroponicMemory: 'Hydroponic Memory',
    engineOfGrowth: 'Engine of Growth'
};

const CHAMBER_TO_SLUG: Record<string, ChamberId> = {
    'archive-of-voices': 'archiveOfVoices',
    'gallery-of-forms': 'galleryOfForms',
    'hydroponic-memory': 'hydroponicMemory',
    'engine-of-growth': 'engineOfGrowth'
};

const STORAGE_KEY = 'vimana-shell-collection';

export class ShellManager {
    private static instance: ShellManager;
    private state: ShellCollectionState;
    private currentShell: ShellCollectible | null = null;

    private constructor() {
        this.state = this.loadState();
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): ShellManager {
        if (!ShellManager.instance) {
            ShellManager.instance = new ShellManager();
        }
        return ShellManager.instance;
    }

    /**
     * Spawn a shell collectible at the given position
     */
    public spawnShell(scene: THREE.Scene, camera: THREE.Camera, position?: THREE.Vector3): ShellCollectible {
        // Remove existing shell if any
        if (this.currentShell) {
            this.currentShell.destroy();
        }

        this.currentShell = new ShellCollectible(scene, camera, {
            spawnPosition: position || new THREE.Vector3(0, 1.0, 1.0)
        });

        return this.currentShell;
    }

    /**
     * Get current shell
     */
    public getCurrentShell(): ShellCollectible | null {
        return this.currentShell;
    }

    /**
     * Mark a shell as collected
     */
    public collectShell(chamberId: string): void {
        const stateKey = CHAMBER_TO_SLUG[chamberId] as ChamberId;
        if (!stateKey || this.state[stateKey]) return;

        this.state[stateKey] = true;
        this.saveState();

        // Dispatch event for UI
        window.dispatchEvent(new CustomEvent('shell-collected', {
            detail: { chamber: chamberId }
        }));
    }

    /**
     * Check if a shell has been collected
     */
    public isCollected(chamberId: string): boolean {
        const stateKey = CHAMBER_TO_SLUG[chamberId] as ChamberId;
        return stateKey ? this.state[stateKey] : false;
    }

    /**
     * Get number of collected shells
     */
    public getCollectedCount(): number {
        return Object.values(this.state).filter(Boolean).length;
    }

    /**
     * Get collection progress (0-1)
     */
    public getProgress(): number {
        return this.getCollectedCount() / 4;
    }

    /**
     * Check if all shells collected
     */
    public isComplete(): boolean {
        return this.getCollectedCount() >= 4;
    }

    /**
     * Get collection state
     */
    public getState(): Readonly<ShellCollectionState> {
        return { ...this.state };
    }

    /**
     * Reset collection state (for testing)
     */
    public reset(): void {
        this.state = {
            archiveOfVoices: false,
            galleryOfForms: false,
            hydroponicMemory: false,
            engineOfGrowth: false
        };
        this.saveState();

        if (this.currentShell) {
            this.currentShell.destroy();
            this.currentShell = null;
        }
    }

    /**
     * Save state to localStorage
     */
    private saveState(): void {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state));
        } catch (e) {
            console.warn('[ShellManager] Failed to save state:', e);
        }
    }

    /**
     * Load state from localStorage
     */
    private loadState(): ShellCollectionState {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                return JSON.parse(saved);
            }
        } catch (e) {
            console.warn('[ShellManager] Failed to load state:', e);
        }

        // Default state
        return {
            archiveOfVoices: false,
            galleryOfForms: false,
            hydroponicMemory: false,
            engineOfGrowth: false
        };
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        if (this.currentShell) {
            this.currentShell.destroy();
            this.currentShell = null;
        }
    }
}
