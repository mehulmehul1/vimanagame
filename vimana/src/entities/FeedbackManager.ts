/**
 * FeedbackManager - Coordinates all feedback systems for gentle guidance
 *
 * Combines camera shake, audio feedback, and visual cues to provide
 * non-punitive feedback when player makes mistakes.
 *
 * Philosophy: "The ship doesn't test you. It teaches you."
 */

import * as THREE from 'three';
import { GentleFeedback } from './GentleFeedback';
import { GentleAudioFeedback } from '../audio/GentleAudioFeedback';
import { NOTE_NAMES, type NoteName } from '../audio/NoteFrequencies';

export interface StringHighlightConfig {
    /** Emissive color for highlighting */
    color: THREE.Color;
    /** Base emissive intensity */
    baseIntensity: number;
    /** Peak emissive intensity during pulse */
    peakIntensity: number;
    /** Pulse duration in seconds */
    pulseDuration: number;
    /** Fade duration in seconds */
    fadeDuration: number;
}

export interface FeedbackManagerConfig {
    /** Enable/disable camera shake */
    enableCameraShake: boolean;
    /** Enable/disable audio feedback */
    enableAudio: boolean;
    /** Enable/disable visual string highlighting */
    enableVisual: boolean;
    /** Custom shake intensity multiplier */
    shakeMultiplier: number;
}

export class FeedbackManager {
    private cameraFeedback: GentleFeedback;
    private audioFeedback: GentleAudioFeedback;
    private scene: THREE.Scene;

    // String highlight state
    private highlightedString: THREE.Mesh | null = null;
    private highlightAnimation: { active: boolean; startTime: number; config: StringHighlightConfig } | null = null;
    private originalEmissive: THREE.Color = new THREE.Color();
    private hasOriginalEmissive: boolean = false;

    // String mesh references (to be populated)
    private stringMeshes: (THREE.Mesh | null)[] = new Array(6).fill(null);

    private config: FeedbackManagerConfig;

    // Warm amber color for highlighting
    private static readonly DEFAULT_HIGHLIGHT_COLOR = new THREE.Color(0xffaa44);

    constructor(
        camera: THREE.Camera,
        scene: THREE.Scene,
        config: Partial<FeedbackManagerConfig> = {}
    ) {
        this.scene = scene;
        this.cameraFeedback = new GentleFeedback(camera);
        this.audioFeedback = new GentleAudioFeedback();

        this.config = {
            enableCameraShake: true,
            enableAudio: true,
            enableVisual: true,
            shakeMultiplier: 1.0,
            ...config
        };

        // Find harp string meshes in scene
        this.findStringMeshes();
    }

    /**
     * Find harp string meshes in the scene
     */
    private findStringMeshes(): void {
        const stringNames = ['String0', 'String1', 'String2', 'String3', 'String4', 'String5'];

        for (let i = 0; i < stringNames.length; i++) {
            const obj = this.scene.getObjectByName(stringNames[i]);
            if (obj && (obj as THREE.Mesh).isMesh) {
                this.stringMeshes[i] = obj as THREE.Mesh;
            }
        }
    }

    /**
     * Refresh string mesh references (call after scene changes)
     */
    public refreshStringMeshes(): void {
        this.findStringMeshes();
    }

    /**
     * Trigger feedback for wrong note played
     * Coordinates camera shake, discordant audio, and visual cue
     *
     * @param targetNoteIndex The correct string index (0-5)
     */
    public async triggerWrongNote(targetNoteIndex: number): Promise<void> {
        const noteName = NOTE_NAMES[targetNoteIndex] as NoteName;

        // Camera shake
        if (this.config.enableCameraShake) {
            this.cameraFeedback.shakeWrongNote();
        }

        // Discordant audio sequence
        if (this.config.enableAudio) {
            this.audioFeedback.playWrongNoteSequence(noteName);
        }

        // Visual cue on correct string
        if (this.config.enableVisual) {
            this.highlightString(targetNoteIndex);
        }
    }

    /**
     * Trigger feedback for premature play (playing before jelly finishes teaching)
     * Uses lighter shake, no audio
     */
    public triggerPrematurePlay(): void {
        if (this.config.enableCameraShake) {
            this.cameraFeedback.shakePremature();
        }

        // No audio for premature - just subtle visual hint
        // The visual hint comes from the jelly continuing its animation
    }

    /**
     * Trigger reminder feedback (gentle nudge)
     */
    public triggerReminder(targetNoteIndex: number): void {
        if (this.config.enableCameraShake) {
            this.cameraFeedback.shakeSubtle();
        }

        if (this.config.enableAudio) {
            const noteName = NOTE_NAMES[targetNoteIndex] as NoteName;
            this.audioFeedback.playReminderTone(noteName);
        }
    }

    /**
     * Play correct note confirmation
     */
    public async triggerCorrectNote(noteIndex: number): Promise<void> {
        if (this.config.enableAudio) {
            const noteName = NOTE_NAMES[noteIndex] as NoteName;
            await this.audioFeedback.playCorrectTone(noteName);
        }
    }

    /**
     * Trigger celebration for completing a sequence
     */
    public triggerSequenceComplete(sequenceIndex: number): void {
        // Multi-shake celebration
        if (this.config.enableCameraShake) {
            this.cameraFeedback.shakeSubtle();
            setTimeout(() => this.cameraFeedback.shakeSubtle(), 500);
        }

        // Highlight all strings briefly? No, just the last one is enough or none
    }

    /**
     * Highlight a specific string with glow animation
     *
     * @param stringIndex The string index to highlight (0-5)
     */
    private highlightString(stringIndex: number): void {
        const stringMesh = this.stringMeshes[stringIndex];
        if (!stringMesh) return;

        const material = stringMesh.material as any;

        // Store original emissive if not already stored
        if (!this.hasOriginalEmissive && material.emissive) {
            this.originalEmissive.copy(material.emissive);
            this.hasOriginalEmissive = true;
        }

        const config: StringHighlightConfig = {
            color: FeedbackManager.DEFAULT_HIGHLIGHT_COLOR,
            baseIntensity: 0.2,
            peakIntensity: 0.5,
            pulseDuration: 1.0,
            fadeDuration: 1.0
        };

        this.highlightedString = stringMesh;
        this.highlightAnimation = {
            active: true,
            startTime: performance.now(),
            config
        };
    }

    /**
     * Update highlight animation - call every frame
     */
    public update(deltaTime: number): void {
        // Update camera shake
        this.cameraFeedback.update(deltaTime);

        // Update string highlight animation
        if (this.highlightAnimation && this.highlightAnimation.active && this.highlightedString) {
            const elapsed = (performance.now() - this.highlightAnimation.startTime) / 1000;
            const config = this.highlightAnimation.config;
            const material = this.highlightedString.material as any;

            if (material.emissive) {
                if (elapsed < config.pulseDuration) {
                    // Pulse phase: base -> peak -> base
                    const pulseProgress = elapsed / config.pulseDuration;
                    const pulse = Math.sin(pulseProgress * Math.PI);
                    const intensity = config.baseIntensity + (config.peakIntensity - config.baseIntensity) * pulse;
                    material.emissive.copy(config.color).multiplyScalar(intensity);
                } else if (elapsed < config.pulseDuration + config.fadeDuration) {
                    // Fade phase
                    const fadeProgress = (elapsed - config.pulseDuration) / config.fadeDuration;
                    const intensity = config.baseIntensity * (1 - fadeProgress);
                    material.emissive.copy(config.color).multiplyScalar(intensity);
                } else {
                    // Animation complete
                    this.endHighlight();
                }
            }
        }
    }

    /**
     * End current highlight animation
     */
    private endHighlight(): void {
        if (this.highlightedString) {
            const material = this.highlightedString.material as any;
            if (material.emissive) {
                if (this.hasOriginalEmissive) {
                    material.emissive.copy(this.originalEmissive);
                } else {
                    material.emissive.set(0, 0, 0);
                }
            }
        }

        this.highlightAnimation = null;
        this.highlightedString = null;
    }

    /**
     * Manually clear any active highlight
     */
    public clearHighlight(): void {
        this.endHighlight();
    }

    /**
     * Set harp string mesh reference manually
     * Use this if string naming differs from default
     */
    public setStringMesh(index: number, mesh: THREE.Mesh): void {
        if (index >= 0 && index < 6) {
            this.stringMeshes[index] = mesh;
        }
    }

    /**
     * Enable or disable camera shake
     */
    public setCameraShakeEnabled(enabled: boolean): void {
        this.config.enableCameraShake = enabled;
        this.cameraFeedback.setEnabled(enabled);
    }

    /**
     * Enable or disable audio feedback
     */
    public setAudioEnabled(enabled: boolean): void {
        this.config.enableAudio = enabled;
        this.audioFeedback.setEnabled(enabled);
    }

    /**
     * Enable or disable visual feedback
     */
    public setVisualEnabled(enabled: boolean): void {
        this.config.enableVisual = enabled;
    }

    /**
     * Update master audio volume
     */
    public setMasterVolume(volume: number): void {
        this.audioFeedback.setMasterVolume(volume);
    }

    /**
     * Update shake intensity multiplier
     */
    public setShakeMultiplier(multiplier: number): void {
        this.config.shakeMultiplier = Math.max(0, Math.min(2, multiplier));

        // Update shake config
        const intensity = this.config.shakeMultiplier;
        this.cameraFeedback.updateConfig({
            maxOffset: 0.5 * intensity,
            amplitudes: {
                x: 0.5 * intensity,
                y: 0.3 * intensity,
                z: 0.2 * intensity
            }
        });
    }

    /**
     * Check if currently shaking
     */
    public isShaking(): boolean {
        return this.cameraFeedback.isActive();
    }

    /**
     * Check if highlighting is active
     */
    public isHighlighting(): boolean {
        return this.highlightAnimation?.active ?? false;
    }

    /**
     * Get audio feedback instance for direct control
     */
    public getAudioFeedback(): GentleAudioFeedback {
        return this.audioFeedback;
    }

    /**
     * Get camera feedback instance for direct control
     */
    public getCameraFeedback(): GentleFeedback {
        return this.cameraFeedback;
    }

    /**
     * Cleanup all feedback systems
     */
    public async destroy(): Promise<void> {
        this.endHighlight();

        this.cameraFeedback.destroy();
        await this.audioFeedback.destroy();

        // Clear references
        this.stringMeshes.fill(null);
    }
}
