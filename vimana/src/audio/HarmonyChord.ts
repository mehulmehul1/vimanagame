/**
 * HarmonyChord - Musical harmony system for duet feedback
 *
 * Plays perfect fifth harmonies when player plays correct notes.
 * Creates the feeling of "the ship joining in" on your music.
 */

import { NOTE_FREQUENCIES, type NoteName } from './NoteFrequencies';

export interface HarmonyConfig {
    /** Harmony interval ratio (default 1.5 = perfect fifth) */
    intervalRatio: number;
    /** Player note volume (0-1) */
    playerVolume: number;
    /** Harmony note volume (0-1) */
    harmonyVolume: number;
    /** Master volume for harmony chords (0-1) */
    masterVolume: number;
    /** Attack time in seconds */
    attackTime: number;
    /** Release time in seconds */
    releaseTime: number;
    /** Enable/disable harmony */
    enabled: boolean;
}

export class HarmonyChord {
    private audioContext: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private config: HarmonyConfig;
    private isInitialized: boolean = false;

    // C Major chord frequencies for completion
    private static readonly C_MAJOR_FREQS = [261.63, 329.63, 392.00]; // C, E, G
    private static readonly C_MAJOR_GAINS = [0.4, 0.35, 0.4];

    constructor(config: Partial<HarmonyConfig> = {}) {
        this.config = {
            intervalRatio: 1.5, // Perfect fifth
            playerVolume: 0.6,
            harmonyVolume: 0.4,
            masterVolume: 0.4,
            attackTime: 0.05,
            releaseTime: 0.3,
            enabled: true,
            ...config
        };
    }

    /**
     * Ensure AudioContext is initialized and resumed
     */
    private async ensureResumed(): Promise<void> {
        if (!this.audioContext) {
            const AudioClass = (window as any).AudioContext || (window as any).webkitAudioContext;
            if (!AudioClass) {
                console.warn('[HarmonyChord] Web Audio API not supported');
                return;
            }
            this.audioContext = new AudioClass();
            if (!this.audioContext) return;

            this.masterGain = this.audioContext.createGain();
            if (this.masterGain) {
                this.masterGain.gain.value = this.config.masterVolume;
                this.masterGain.connect(this.audioContext.destination);
            }
            this.isInitialized = true;
            console.log('[HarmonyChord] AudioContext created, state:', this.audioContext.state);
        }

        if (this.audioContext && this.audioContext.state === 'suspended') {
            console.log('[HarmonyChord] Resuming suspended AudioContext...');
            await this.audioContext.resume();
            console.log('[HarmonyChord] AudioContext resumed, state:', this.audioContext.state);
        }

        if (!this.audioContext || !this.masterGain) {
            throw new Error('Failed to initialize AudioContext or masterGain');
        }
    }

    /**
     * Public method to resume AudioContext (called on user interaction)
     * Browsers require user interaction before audio can play
     */
    public async resume(): Promise<void> {
        await this.ensureResumed();
    }

    /**
     * Play demonstration note
     * Used when jelly demonstrates which note to play
     *
     * @param noteIndex The note index (0-5)
     * @param duration Duration in seconds
     */
    public async playDemonstrationNote(noteIndex: number, duration: number = 0.5): Promise<void> {
        console.log('[HarmonyChord] playDemonstrationNote:', noteIndex, 'duration:', duration, 'enabled:', this.config.enabled);

        if (!this.config.enabled) {
            console.warn('[HarmonyChord] Audio is disabled!');
            return;
        }

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) {
            console.warn('[HarmonyChord] AudioContext or masterGain not available');
            return;
        }

        // Validate note index
        if (noteIndex < 0 || noteIndex > 5 || isNaN(noteIndex)) {
            console.warn('[HarmonyChord] Invalid note index:', noteIndex);
            return;
        }

        const noteName = ['C', 'D', 'E', 'F', 'G', 'A'][noteIndex];
        const freq = NOTE_FREQUENCIES[noteName as NoteName];

        if (!freq || !isFinite(freq)) {
            console.warn('[HarmonyChord] Invalid frequency for note:', noteName, 'index:', noteIndex);
            return;
        }

        console.log('[HarmonyChord] Playing note:', noteName, 'freq:', freq);

        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        osc.frequency.value = freq;
        osc.type = 'sine';

        osc.connect(gain);
        gain.connect(this.masterGain);

        const now = this.audioContext.currentTime;
        const totalDuration = this.config.attackTime + duration + this.config.releaseTime;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(this.config.masterVolume * 0.5, now + this.config.attackTime);
        gain.gain.linearRampToValueAtTime(0, now + totalDuration);

        osc.start(now);
        osc.stop(now + totalDuration);
    }

    /**
     * Play harmony chord when player plays correct note
     * Plays player note + perfect fifth harmony
     *
     * @param noteIndex The note index (0-5 for C-A)
     * @param duration Duration in seconds (default 0.55)
     */
    public async playHarmony(noteIndex: number, duration: number = 0.55): Promise<void> {
        console.log('[HarmonyChord] playHarmony:', noteIndex, 'duration:', duration);

        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const noteName = ['C', 'D', 'E', 'F', 'G', 'A'][noteIndex] as NoteName;
        const baseFreq = NOTE_FREQUENCIES[noteName];
        const harmonyFreq = baseFreq * this.config.intervalRatio;
        console.log('[HarmonyChord] Playing harmony for:', noteName, 'baseFreq:', baseFreq, 'harmonyFreq:', harmonyFreq);

        // Create oscillators
        const playerOsc = this.audioContext.createOscillator();
        const playerGain = this.audioContext.createGain();
        const harmonyOsc = this.audioContext.createOscillator();
        const harmonyGain = this.audioContext.createGain();

        // Set frequencies
        playerOsc.frequency.value = baseFreq;
        playerOsc.type = 'sine';
        harmonyOsc.frequency.value = harmonyFreq;
        harmonyOsc.type = 'sine';

        // Connect graph
        playerOsc.connect(playerGain);
        playerGain.connect(this.masterGain);
        harmonyOsc.connect(harmonyGain);
        harmonyGain.connect(this.masterGain);

        // Envelope
        const now = this.audioContext.currentTime;
        const totalDuration = this.config.attackTime + duration + this.config.releaseTime;

        // Player note envelope
        playerGain.gain.setValueAtTime(0, now);
        playerGain.gain.linearRampToValueAtTime(this.config.playerVolume, now + this.config.attackTime);
        playerGain.gain.linearRampToValueAtTime(0, now + totalDuration);

        // Harmony note envelope (slightly softer, doesn't overpower)
        harmonyGain.gain.setValueAtTime(0, now);
        harmonyGain.gain.linearRampToValueAtTime(this.config.harmonyVolume, now + this.config.attackTime);
        harmonyGain.gain.linearRampToValueAtTime(0, now + totalDuration);

        // Start and stop
        playerOsc.start(now);
        harmonyOsc.start(now);
        playerOsc.stop(now + totalDuration);
        harmonyOsc.stop(now + totalDuration);
    }

    /**
     * Play individual note confirmation (subtle feedback)
     * Used during phrase-first response phase
     *
     * @param noteIndex The note index (0-5)
     */
    public async playNoteConfirmation(noteIndex: number): Promise<void> {
        console.log('[HarmonyChord] playNoteConfirmation:', noteIndex);

        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const noteName = ['C', 'D', 'E', 'F', 'G', 'A'][noteIndex] as NoteName;
        const freq = NOTE_FREQUENCIES[noteName];

        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        // Higher harmonic for subtle confirmation feel
        osc.frequency.value = freq * 2.0;
        osc.type = 'sine';

        osc.connect(gain);
        gain.connect(this.masterGain);

        const now = this.audioContext.currentTime;
        const duration = 0.15; // Short duration

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(this.config.harmonyVolume * 0.5, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

        osc.start(now);
        osc.stop(now + duration);
    }

    /**
     * Play single demonstration note (DEPRECATED - use playDemonstrationNote)
     * Used when jelly demonstrates which note to play
     *
     * @param noteIndex The note index (0-5)
     * @param duration Duration in seconds
     */
    public async playDemonstrationNote_OLD(noteIndex: number, duration: number = 0.5): Promise<void> {
        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const noteName = ['C', 'D', 'E', 'F', 'G', 'A'][noteIndex] as NoteName;
        const freq = NOTE_FREQUENCIES[noteName];

        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        osc.frequency.value = freq;
        osc.type = 'sine';

        osc.connect(gain);
        gain.connect(this.masterGain);

        const now = this.audioContext.currentTime;
        const totalDuration = this.config.attackTime + duration + this.config.releaseTime;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(this.config.masterVolume * 0.5, now + this.config.attackTime);
        gain.gain.linearRampToValueAtTime(0, now + totalDuration);

        osc.start(now);
        osc.stop(now + totalDuration);
    }

    /**
     * Update master volume
     */
    public setMasterVolume(volume: number): void {
        this.config.masterVolume = Math.max(0, Math.min(1, volume));
        if (this.masterGain && this.audioContext) {
            this.masterGain.gain.setValueAtTime(
                this.config.masterVolume,
                this.audioContext.currentTime
            );
        }
    }

    /**
     * Enable or disable harmony
     */
    public setEnabled(enabled: boolean): void {
        this.config.enabled = enabled;
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<HarmonyConfig>): void {
        this.config = { ...this.config, ...updates };
    }

    /**
     * Get current configuration
     */
    public getConfig(): Readonly<HarmonyConfig> {
        return { ...this.config };
    }

    /**
     * Play splash sound for synchronized landing (STORY-HARP-102)
     * Single unified splash sound, not multiple overlapping splashes
     */
    public async playSplashSound(): Promise<void> {
        console.log('[HarmonyChord] Playing unified splash sound');

        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const now = this.audioContext.currentTime;
        const duration = 0.8; // 800ms splash duration

        // Create pink noise buffer for splash sound
        const splashBuffer = this.createSplashBuffer(duration);

        // Create source for the splash
        const source = this.audioContext.createBufferSource();
        source.buffer = splashBuffer;

        // Create gain node for envelope
        const gainNode = this.audioContext.createGain();

        // Connect: source -> gain -> master
        source.connect(gainNode);
        gainNode.connect(this.masterGain);

        // Envelope: quick attack, exponential decay
        const attackTime = 0.05;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.3, now + attackTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, now + duration);

        // Start and stop
        source.start(now);
        source.stop(now + duration);
    }

    /**
     * Generate splash sound buffer (STORY-HARP-102)
     * Pink noise with exponential envelope for water splash effect
     */
    private createSplashBuffer(duration: number): AudioBuffer {
        const sampleRate = this.audioContext!.sampleRate;
        const buffer = this.audioContext!.createBuffer(1, sampleRate * duration, sampleRate);
        const data = buffer.getChannelData(0);

        // Generate pink noise-like sound with exponential decay
        for (let i = 0; i < data.length; i++) {
            const t = i / sampleRate;
            // Envelope: quick attack, exponential decay
            const envelope = Math.exp(-t * 5);
            // Noise with lowpass characteristic (water-like)
            const noise = (Math.random() * 2 - 1) * envelope * 0.3;
            data[i] = noise;
        }

        return buffer;
    }

    /**
     * Play completion chord (C Major triad)
     * Called when a sequence is completed
     *
     * @param duration Duration in seconds (default 1.0)
     */
    public async playCompletionChord(duration: number = 1.0): Promise<void> {
        console.log('[HarmonyChord] playCompletionChord: duration:', duration, 'enabled:', this.config.enabled);

        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        console.log('[HarmonyChord] Playing C Major chord:', HarmonyChord.C_MAJOR_FREQS);

        const now = this.audioContext.currentTime;
        const totalDuration = this.config.attackTime + duration + this.config.releaseTime;

        // Create oscillators for C Major chord (C, E, G)
        const oscillators: OscillatorNode[] = [];
        const gains: GainNode[] = [];

        for (let i = 0; i < HarmonyChord.C_MAJOR_FREQS.length; i++) {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            osc.frequency.value = HarmonyChord.C_MAJOR_FREQS[i];
            osc.type = 'sine';

            osc.connect(gain);
            gain.connect(this.masterGain);

            // Envelope with individual gains
            gain.gain.setValueAtTime(0, now);
            gain.gain.linearRampToValueAtTime(
                HarmonyChord.C_MAJOR_GAINS[i],
                now + this.config.attackTime
            );
            gain.gain.linearRampToValueAtTime(0, now + totalDuration);

            osc.start(now);
            osc.stop(now + totalDuration);

            oscillators.push(osc);
            gains.push(gain);
        }
    }

    /**
     * Check if ready to play
     */
    public isReady(): boolean {
        return this.isInitialized && this.audioContext !== null;
    }

    /**
     * Cleanup and close audio context
     */
    public async destroy(): Promise<void> {
        if (this.audioContext && this.audioContext.state !== 'closed') {
            try {
                await this.audioContext.close();
            } catch (e) {
                // Context may already be closed
            }
        }
        this.audioContext = null;
        this.masterGain = null;
        this.isInitialized = false;
    }
}
