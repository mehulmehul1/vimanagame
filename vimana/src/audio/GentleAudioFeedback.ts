/**
 * GentleAudioFeedback - Web Audio API based sound feedback
 *
 * Provides discordant tones for wrong notes and gentle reminder tones.
 * Uses pure synthesis (no external audio files).
 */

import { NOTE_FREQUENCIES, REMINDER_FREQUENCY, getDiscordantFrequency, type NoteName } from './NoteFrequencies';

export interface AudioFeedbackConfig {
    /** Master volume (0-1) */
    masterVolume: number;
    /** Discordant tone volume multiplier */
    discordantVolume: number;
    /** Reminder tone volume multiplier */
    reminderVolume: number;
    /** Enable/disable audio */
    enabled: boolean;
}

export class GentleAudioFeedback {
    private audioContext: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private config: AudioFeedbackConfig;
    private isInitialized: boolean = false;

    constructor(config: Partial<AudioFeedbackConfig> = {}) {
        this.config = {
            masterVolume: 0.3,
            discordantVolume: 0.5,
            reminderVolume: 0.25,
            enabled: true,
            ...config
        };
    }

    /**
     * Ensure AudioContext is initialized and resumed
     * Browsers require user interaction before audio can play
     */
    private async ensureResumed(): Promise<void> {
        if (!this.audioContext) {
            const AudioClass = (window as any).AudioContext || (window as any).webkitAudioContext;
            if (!AudioClass) {
                console.warn('Web Audio API not supported');
                return;
            }
            this.audioContext = new AudioClass();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = this.config.masterVolume;
            this.masterGain.connect(this.audioContext.destination);
            this.isInitialized = true;
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
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
     * Play a discordant tone when player plays wrong note
     * Uses minor second dissonance for "not quite right" feeling
     *
     * @param correctNote The note that should have been played
     * @param duration Duration of the tone in seconds
     */
    public async playDiscordantTone(correctNote: NoteName, duration: number = 0.3): Promise<void> {
        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const correctFreq = NOTE_FREQUENCIES[correctNote];
        if (!correctFreq || !isFinite(correctFreq)) {
            console.warn('[GentleAudioFeedback] Invalid note frequency for:', correctNote);
            return;
        }
        const dissonantFreq = getDiscordantFrequency(correctFreq, true);

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        // Use sine wave with slight vibrato for organic dissonance
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(dissonantFreq, this.audioContext.currentTime);

        // Add slight detune for more organic dissonance (+5 cents)
        oscillator.detune.setValueAtTime(5, this.audioContext.currentTime);

        // Add gentle vibrato (3-5 Hz) using a second oscillator
        const vibratoOsc = this.audioContext.createOscillator();
        const vibratoGain = this.audioContext.createGain();
        vibratoOsc.type = 'sine';
        vibratoOsc.frequency.setValueAtTime(4, this.audioContext.currentTime); // 4 Hz vibrato
        vibratoGain.gain.setValueAtTime(3, this.audioContext.currentTime); // ±3 cents
        vibratoOsc.connect(vibratoGain);
        vibratoGain.connect(oscillator.frequency);
        vibratoOsc.start(this.audioContext.currentTime);
        vibratoOsc.stop(this.audioContext.currentTime + duration);

        // Gain envelope: smooth attack and decay
        const now = this.audioContext.currentTime;
        const attackTime = 0.05;
        const maxGain = this.config.discordantVolume;

        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(maxGain, now + attackTime);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }

    /**
     * Play a gentle reminder tone after discordant sound
     * Indicates "try again, listen to this note"
     *
     * @param targetNote The note the player should try
     * @param duration Duration of the tone in seconds
     */
    public async playReminderTone(targetNote: NoteName, duration: number = 0.2): Promise<void> {
        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        // Use E4 as gentle reminder (or target note)
        const reminderFreq = REMINDER_FREQUENCY;

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        // Pure sine wave for gentleness
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(reminderFreq, this.audioContext.currentTime);

        // Soft attack and decay
        const now = this.audioContext.currentTime;
        const attackTime = 0.03;
        const maxGain = this.config.reminderVolume;

        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(maxGain, now + attackTime);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }

    /**
     * Play complete wrong note feedback sequence
     * Discordant tone -> gap -> reminder tone
     *
     * @param correctNote The correct note that should have been played
     */
    public async playWrongNoteSequence(correctNote: NoteName): Promise<void> {
        if (!this.config.enabled) return;

        // Play discordant tone
        await this.playDiscordantTone(correctNote, 0.3);

        // Wait for gap (discordant duration + 100ms)
        await new Promise(resolve => setTimeout(resolve, 400));

        // Play reminder tone
        await this.playReminderTone(correctNote, 0.2);
    }

    /**
     * Play a gentle confirmation tone for correct note
     * Subtle positive feedback
     *
     * @param note The note played correctly
     */
    public async playCorrectTone(note: NoteName, duration: number = 0.15): Promise<void> {
        if (!this.config.enabled) return;

        await this.ensureResumed();
        if (!this.audioContext || !this.masterGain) return;

        const freq = NOTE_FREQUENCIES[note];

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(freq, this.audioContext.currentTime);

        const now = this.audioContext.currentTime;
        const maxGain = this.config.reminderVolume * 0.7; // Slightly quieter

        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(maxGain, now + 0.02);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }

    /**
     * Update master volume
     */
    public setMasterVolume(volume: number): void {
        this.config.masterVolume = Math.max(0, Math.min(1, volume));
        if (this.masterGain) {
            this.masterGain.gain.setValueAtTime(
                this.config.masterVolume,
                this.audioContext?.currentTime || 0
            );
        }
    }

    /**
     * Enable or disable audio feedback
     */
    public setEnabled(enabled: boolean): void {
        this.config.enabled = enabled;
    }

    /**
     * Check if audio is enabled
     */
    public isEnabled(): boolean {
        return this.config.enabled;
    }

    /**
     * Get current configuration
     */
    public getConfig(): Readonly<AudioFeedbackConfig> {
        return { ...this.config };
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<AudioFeedbackConfig>): void {
        this.config = { ...this.config, ...updates };

        // Apply volume changes immediately
        if (updates.masterVolume !== undefined && this.masterGain && this.audioContext) {
            this.masterGain.gain.setValueAtTime(
                this.config.masterVolume,
                this.audioContext.currentTime
            );
        }
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

    /**
     * Check if audio system is initialized
     */
    public isReady(): boolean {
        return this.isInitialized && this.audioContext !== null;
    }
}
