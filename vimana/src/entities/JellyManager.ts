import * as THREE from 'three';
import { JellyCreature } from './JellyCreature';

/**
 * JellyManager - Manages the jelly creature choir
 *
 * Coordinates multiple jelly creatures for teaching demonstrations.
 */
export class JellyManager extends THREE.Group {
    private jellies: JellyCreature[] = [];
    private activeJelly: JellyCreature | null = null;
    // STORY-HARP-101: Support multiple active jellies for phrase-first mode
    private activeJellies: Set<JellyCreature> = new Set();

    // Story 1.2 spec: X offsets for 6 strings, Z is negative (toward camera)
    private static readonly JELLY_X_OFFSETS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
    private static readonly JELLY_Z_OFFSET = -1.0; // Negative = toward camera

    // Story 1.2 spec: Pulse rates C=1.0, D=1.1, E=1.2, F=1.3, G=1.4, A=1.5
    private static readonly PULSE_RATES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5];

    // Water surface Y (set dynamically by HarpRoom)
    private waterSurfaceY: number = 0.0;

    constructor() {
        super();
        this.createJellies();
    }

    private createJellies(): void {
        for (let i = 0; i < 6; i++) {
            // Position at water level with correct Z direction
            const pos = new THREE.Vector3(
                JellyManager.JELLY_X_OFFSETS[i],
                0, // Y will be set by updateWaterSurface()
                JellyManager.JELLY_Z_OFFSET
            );
            const jelly = new JellyCreature(pos, i);
            jelly.setPulseRate(JellyManager.PULSE_RATES[i]);

            // Story spec: Unique color per note (cyan-blue range)
            // Using HSL: base hue 0.55 (cyan-blue) with variation
            const hue = 0.5 + (i / 12.0); // 0.5 to 0.55 range
            const color = new THREE.Color().setHSL(hue, 0.7, 0.6);
            jelly.setColor(color);

            this.jellies.push(jelly);
            this.add(jelly);
        }
    }

    /**
     * Update jelly positions based on actual water surface Y
     * Called by HarpRoom after detecting ArenaFloor position
     */
    public updateWaterSurface(waterSurfaceY: number): void {
        this.waterSurfaceY = waterSurfaceY;

        for (let i = 0; i < this.jellies.length; i++) {
            const jelly = this.jellies[i];
            const homePos = new THREE.Vector3(
                JellyManager.JELLY_X_OFFSETS[i],
                waterSurfaceY, // Water surface level
                JellyManager.JELLY_Z_OFFSET
            );
            jelly.setHomePosition(homePos);
            // Also set water surface for swim physics
            jelly.setWaterSurface(waterSurfaceY);
        }
        console.log(`[JellyManager] ✅ Updated positions for water surface Y: ${waterSurfaceY.toFixed(3)}`);
    }

    /**
     * Spawn a jelly to demonstrate a specific string
     * STORY-HARP-101: Now supports multiple active jellies
     */
    public spawnJelly(stringIndex: number): void {
        if (stringIndex >= 0 && stringIndex < this.jellies.length) {
            const jelly = this.jellies[stringIndex];
            if (jelly.isHidden()) {
                jelly.spawn(stringIndex);
                this.activeJelly = jelly;
                // STORY-HARP-101: Track multiple active jellies
                this.activeJellies.add(jelly);
            }
        }
    }

    /**
     * Make active jelly begin teaching
     */
    public beginTeaching(): void {
        if (this.activeJelly) {
            this.activeJelly.beginTeaching();
        }
    }

    /**
     * Submerge active jelly (note-by-note mode)
     * STORY-HARP-101: For phrase-first, use submergeAll() instead
     */
    public submergeActive(): void {
        if (this.activeJelly) {
            this.activeJelly.submerge();
            this.activeJellies.delete(this.activeJelly);
            this.activeJelly = null;
        }
    }

    /**
     * Submerge all active jellies (STORY-HARP-101)
     * Used for synchronized splash turn signal
     */
    public submergeAll(): void {
        for (const jelly of this.activeJellies) {
            jelly.submerge();
        }
        this.activeJellies.clear();
        this.activeJelly = null;
    }

    /**
     * Get jelly by string index
     */
    public getJelly(stringIndex: number): JellyCreature | undefined {
        return this.jellies[stringIndex];
    }

    /**
     * Update all jellies
     */
    public update(deltaTime: number, time: number, cameraPosition: THREE.Vector3): void {
        for (const jelly of this.jellies) {
            jelly.setCameraPosition(cameraPosition);
            jelly.update(deltaTime, time);
        }
    }

    /**
     * Get active teaching jelly
     */
    public getActiveJelly(): JellyCreature | null {
        return this.activeJelly;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        for (const jelly of this.jellies) {
            jelly.destroy();
            this.remove(jelly);
        }
        this.jellies = [];
        this.activeJelly = null;
        // STORY-HARP-101: Clear active jellies set
        this.activeJellies.clear();
    }
}
