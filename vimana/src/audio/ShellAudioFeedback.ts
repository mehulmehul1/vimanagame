/**
 * ShellAudioFeedback - Audio feedback for shell interactions
 *
 * Provides sounds for shell spawn, hover, and collection.
 */

export class ShellAudioFeedback {
    private audioContext: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private enabled: boolean = true;
    private volume: number = 0.3;

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
                console.warn('[ShellAudioFeedback] Web Audio API not supported');
                return;
            }
            this.audioContext = new AudioClass();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = this.volume;
            this.masterGain.connect(this.audioContext.destination);
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    /**
     * Play spawn sound (gentle chime)
     */
    public async playSpawnSound(): Promise<void> {
        if (!this.enabled) return;
        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        // E5 + G5 harmony
        const freqs = [659.25, 783.99]; // E5, G5
        const now = this.audioContext.currentTime;
        const duration = 0.8;

        for (let i = 0; i < freqs.length; i++) {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            osc.type = 'sine';
            osc.frequency.value = freqs[i];

            gain.gain.setValueAtTime(0, now);
            gain.gain.linearRampToValueAtTime(0.2, now + 0.1);
            gain.gain.linearRampToValueAtTime(0, now + duration);

            osc.connect(gain);
            gain.connect(this.masterGain);

            osc.start(now);
            osc.stop(now + duration);
        }
    }

    /**
     * Play hover sound (subtle sparkle)
     */
    public async playHoverSound(): Promise<void> {
        if (!this.enabled) return;
        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        osc.type = 'sine';
        osc.frequency.value = 1318.51; // E6

        const now = this.audioContext.currentTime;
        const duration = 0.15;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.1, now + 0.02);
        gain.gain.linearRampToValueAtTime(0, now + duration);

        osc.connect(gain);
        gain.connect(this.masterGain);

        osc.start(now);
        osc.stop(now + duration);
    }

    /**
     * Play collection sound (ascending arpeggio)
     */
    public async playCollectSound(): Promise<void> {
        if (!this.enabled) return;
        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        // C5 → E5 → G5 → C6 arpeggio
        const freqs = [523.25, 659.25, 783.99, 1046.50];
        const now = this.audioContext.currentTime;
        const duration = 1.0;

        for (let i = 0; i < freqs.length; i++) {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            osc.type = 'sine';
            osc.frequency.value = freqs[i];

            const offset = i * 0.1;
            const noteDuration = duration - offset;

            gain.gain.setValueAtTime(0, now + offset);
            gain.gain.linearRampToValueAtTime(0.2, now + offset + 0.05);
            gain.gain.linearRampToValueAtTime(0, now + offset + noteDuration);

            osc.connect(gain);
            gain.connect(this.masterGain);

            osc.start(now);
            osc.stop(now + offset + noteDuration);
        }
    }

    /**
     * Set master volume
     */
    public setVolume(volume: number): void {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.masterGain && this.audioContext) {
            this.masterGain.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
        }
    }

    /**
     * Enable or disable audio
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;
    }

    /**
     * Cleanup
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
    }
}
