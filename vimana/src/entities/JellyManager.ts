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

    // 6 jellies for 6 strings (C, D, E, F, G, A)
    private static readonly JELLY_POSITIONS = [
        new THREE.Vector3(-2.5, 0, 1),
        new THREE.Vector3(-1.5, 0, 1),
        new THREE.Vector3(-0.5, 0, 1),
        new THREE.Vector3(0.5, 0, 1),
        new THREE.Vector3(1.5, 0, 1),
        new THREE.Vector3(2.5, 0, 1)
    ];

    // Pulse rates for each note (visual variety)
    private static readonly PULSE_RATES = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8];

    constructor() {
        super();
        this.createJellies();
    }

    private createJellies(): void {
        for (let i = 0; i < 6; i++) {
            const jelly = new JellyCreature(JellyManager.JELLY_POSITIONS[i]);
            jelly.setPulseRate(JellyManager.PULSE_RATES[i]);

            // Different color for each string position
            const hue = i / 6.0;
            const color = new THREE.Color().setHSL(0.4 + hue * 0.2, 0.8, 0.6);
            jelly.setColor(color);

            this.jellies.push(jelly);
            this.add(jelly);
        }
    }

    /**
     * Spawn a jelly to demonstrate a specific string
     */
    public spawnJelly(stringIndex: number): void {
        if (stringIndex >= 0 && stringIndex < this.jellies.length) {
            const jelly = this.jellies[stringIndex];
            if (jelly.isHidden()) {
                jelly.spawn(stringIndex);
                this.activeJelly = jelly;
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
     * Submerge active jelly
     */
    public submergeActive(): void {
        if (this.activeJelly) {
            this.activeJelly.submerge();
            this.activeJelly = null;
        }
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
     * Get jelly by string index
     */
    public getJelly(stringIndex: number): JellyCreature | undefined {
        return this.jellies[stringIndex];
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
    }
}
