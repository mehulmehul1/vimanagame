/**
 * WhiteFlashAudio - Transcendent audio for white flash ending
 *
 * Creates an ethereal soundscape that swells during the
 * white flash transition, symbolizing transcendence.
 */

export class WhiteFlashAudio {
    private audioContext: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private enabled: boolean = true;

    // Oscillator nodes for the drone
    private oscillators: OscillatorNode[] = [];
    private gains: GainNode[] = [];

    // Envelope
    private masterEnvelope: GainNode | null = null;

    // Configuration
    private readonly baseFreq = 261.63; // Middle C
    private readonly harmonyFreqs = [1.0, 1.5, 2.0, 2.5]; // Unison, fifth, octave, third
    private readonly maxVolume = 0.4;

    // State
    private active: boolean = false;
    private crescendoProgress: number = 0;
    private fadeProgress: number = 0;

    constructor() {
        // Initialize on first use
    }

    /**
     * Ensure AudioContext is ready
     */
    private async ensureResumed(): Promise<void> {
        if (!this.audioContext) {
            const AudioClass = (window as any).AudioContext || (window as any).webkitAudioContext;
            if (!AudioClass) {
                console.warn('[WhiteFlashAudio] Web Audio API not supported');
                return;
            }
            this.audioContext = new AudioClass();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = this.maxVolume;
            this.masterGain.connect(this.audioContext.destination);

            // Create master envelope
            this.masterEnvelope = this.audioContext.createGain();
            this.masterEnvelope.gain.value = 0;
            this.masterEnvelope.connect(this.masterGain);
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    /**
     * Start the crescendo
     */
    public async startCrescendo(): Promise<void> {
        if (!this.enabled || this.active) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterEnvelope) return;

        this.active = true;
        this.crescendoProgress = 0;
        this.fadeProgress = 0;

        // Create harmonic oscillators
        this.createDrone();

        // Start crescendo
        this.masterEnvelope.gain.setValueAtTime(0, this.audioContext.currentTime);
        this.masterEnvelope.gain.linearRampToValueAtTime(
            this.maxVolume,
            this.audioContext.currentTime + 2.0
        );
    }

    /**
     * Create the harmonic drone
     */
    private createDrone(): void {
        if (!this.audioContext || !this.masterEnvelope) return;

        this.harmonyFreqs.forEach((ratio, index) => {
            const osc = this.audioContext!.createOscillator();
            const gain = this.audioContext!.createGain();

            osc.type = 'sine';
            osc.frequency.value = this.baseFreq * ratio;

            // Volume for each harmonic (fundamental louder)
            const volume = index === 0 ? 0.5 : 0.2 / index;
            gain.gain.value = volume;

            osc.connect(gain);
            gain.connect(this.masterEnvelope!);

            osc.start();

            this.oscillators.push(osc);
            this.gains.push(gain);
        });

        // Add subtle vibrato to fundamental
        this.addVibrato();
    }

    /**
     * Add subtle vibrato for warmth
     */
    private addVibrato(): void {
        if (!this.audioContext || this.oscillators.length === 0) return;

        const lfo = this.audioContext.createOscillator();
        const lfoGain = this.audioContext.createGain();

        lfo.type = 'sine';
        lfo.frequency.value = 3.0; // 3 Hz vibrato
        lfoGain.gain.value = 3.0; // ±3 Hz depth

        lfo.connect(lfoGain);
        lfoGain.connect(this.oscillators[0].frequency);

        lfo.start();

        this.oscillators.push(lfo);
        this.gains.push(lfoGain);
    }

    /**
     * Start fade out
     */
    public startFade(): void {
        if (!this.active || !this.audioContext || !this.masterEnvelope) return;

        const now = this.audioContext.currentTime;
        this.masterEnvelope.gain.cancelScheduledValues(now);
        this.masterEnvelope.gain.setValueAtTime(this.masterEnvelope.gain.value, now);
        this.masterEnvelope.gain.linearRampToValueAtTime(0, now + 3.0);
    }

    /**
     * Update audio state
     */
    public update(deltaTime: number): void {
        if (!this.active) return;

        this.crescendoProgress += deltaTime;

        // Trigger fade after 6 seconds
        if (this.crescendoProgress > 6.0 && this.fadeProgress === 0) {
            this.fadeProgress = 1;
            this.startFade();
        }
    }

    /**
     * Stop all audio immediately
     */
    public stop(): void {
        if (!this.audioContext || !this.masterEnvelope) return;

        const now = this.audioContext.currentTime;
        this.masterEnvelope.gain.cancelScheduledValues(now);
        this.masterEnvelope.gain.setValueAtTime(0, now);

        this.active = false;
    }

    /**
     * Reset to initial state
     */
    public reset(): void {
        this.stop();
        this.crescendoProgress = 0;
        this.fadeProgress = 0;

        // Recreate oscillators on next start
        this.destroyOscillators();
    }

    /**
     * Destroy all oscillator nodes
     */
    private destroyOscillators(): void {
        this.oscillators.forEach(osc => {
            try {
                osc.stop();
            } catch (e) {
                // May already be stopped
            }
        });
        this.oscillators = [];
        this.gains = [];
    }

    /**
     * Enable or disable audio
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        if (!enabled) {
            this.stop();
        }
    }

    /**
     * Set master volume
     */
    public setVolume(volume: number): void {
        const clamped = Math.max(0, Math.min(1, volume));
        if (this.masterGain) {
            this.masterGain.gain.value = clamped * this.maxVolume;
        }
    }

    /**
     * Check if currently active
     */
    public isActive(): boolean {
        return this.active;
    }

    /**
     * Get crescendo progress (0-1)
     */
    public getProgress(): number {
        return Math.min(this.crescendoProgress / 6.0, 1.0);
    }

    /**
     * Cleanup
     */
    public async destroy(): Promise<void> {
        this.reset();
        this.destroyOscillators();

        if (this.audioContext && this.audioContext.state !== 'closed') {
            try {
                await this.audioContext.close();
            } catch (e) {
                // Context may already be closed
            }
        }
        this.audioContext = null;
        this.masterGain = null;
        this.masterEnvelope = null;
    }
}
