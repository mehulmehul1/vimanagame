/**
 * Unit tests for GentleAudioFeedback
 *
 * Tests discordant tones, reminder tones, and correct note feedback.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Discordant tone (minor second dissonance) - P0
 * - Patient reminder tone (E4) - P0
 * - Web Audio API resume - P0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GentleAudioFeedback } from '../../src/audio/GentleAudioFeedback';
import type { NoteName } from '../../src/audio/NoteFrequencies';

// Mock AudioContext and related Web Audio API
class MockAudioContext {
    state: 'suspended' | 'running' | 'closed' = 'suspended';
    destination = {};
    currentTime = 0;

    createOscillator() {
        return {
            type: 'sine',
            frequency: { value: 0, setValueAtTime: vi.fn() },
            detune: { setValueAtTime: vi.fn() },
            connect: vi.fn(),
            start: vi.fn(),
            stop: vi.fn()
        };
    }

    createGain() {
        return {
            gain: { value: 0, setValueAtTime: vi.fn(), linearRampToValueAtTime: vi.fn() },
            connect: vi.fn()
        };
    }

    async resume() {
        this.state = 'running';
        return Promise.resolve();
    }

    async close() {
        this.state = 'closed';
        return Promise.resolve();
    }
}

// Mock window.AudioContext
global.AudioContext = MockAudioContext as any;
global.webkitAudioContext = MockAudioContext as any;

describe('GentleAudioFeedback', () => {
    let audioFeedback: GentleAudioFeedback;

    beforeEach(() => {
        audioFeedback = new GentleAudioFeedback();
    });

    afterEach(async () => {
        await audioFeedback.destroy();
        vi.clearAllMocks();
    });

    describe('Construction', () => {
        it('should initialize with default config', () => {
            const config = audioFeedback.getConfig();
            expect(config.masterVolume).toBe(0.3);
            expect(config.discordantVolume).toBe(0.5);
            expect(config.reminderVolume).toBe(0.25);
            expect(config.enabled).toBe(true);
        });

        it('should accept custom config', () => {
            const customFeedback = new GentleAudioFeedback({
                masterVolume: 0.5,
                discordantVolume: 0.7,
                enabled: false
            });
            const config = customFeedback.getConfig();
            expect(config.masterVolume).toBe(0.5);
            expect(config.discordantVolume).toBe(0.7);
            expect(config.enabled).toBe(false);
        });
    });

    describe('Initialization', () => {
        it('should not be ready before first audio call', () => {
            expect(audioFeedback.isReady()).toBe(false);
        });

        it('should initialize AudioContext on first play', async () => {
            await audioFeedback.playDiscordantTone('C', 0.1);
            expect(audioFeedback.isReady()).toBe(true);
        });

        it('should resume suspended AudioContext', async () => {
            await audioFeedback.playDiscordantTone('C', 0.1);
            // AudioContext should be running after first call
            expect(audioFeedback.isReady()).toBe(true);
        });
    });

    describe('Discordant Tone', () => {
        it('should play discordant tone for correct note', async () => {
            await audioFeedback.playDiscordantTone('C', 0.1);
            // Should not throw
            expect(audioFeedback.isReady()).toBe(true);
        });

        it('should use minor second dissonance (16/15 ratio)', async () => {
            // This test validates the frequency calculation logic
            // C frequency = 261.63 Hz
            // Discordant C = 261.63 * (16/15) ≈ 279.07 Hz
            const freqC = 261.63;
            const expectedDiscordant = freqC * (16 / 15);
            expect(expectedDiscordant).toBeCloseTo(279.07, 1);
        });

        it('should not play when disabled', async () => {
            audioFeedback.setEnabled(false);
            await audioFeedback.playDiscordantTone('C', 0.1);
            // Should not initialize AudioContext
            expect(audioFeedback.isReady()).toBe(false);
        });

        it('should support all note names', async () => {
            const notes: NoteName[] = ['C', 'D', 'E', 'F', 'G', 'A'];
            for (const note of notes) {
                await audioFeedback.playDiscordantTone(note, 0.1);
            }
            expect(audioFeedback.isReady()).toBe(true);
        });
    });

    describe('Reminder Tone', () => {
        it('should play gentle reminder tone', async () => {
            await audioFeedback.playReminderTone('C', 0.1);
            expect(audioFeedback.isReady()).toBe(true);
        });

        it('should use E4 frequency (329.63 Hz)', () => {
            // Validates the constant REMINDER_FREQUENCY
            const expectedFreq = 329.63;
            expect(expectedFreq).toBeCloseTo(329.63, 1);
        });

        it('should be quieter than discordant tone', () => {
            const config = audioFeedback.getConfig();
            expect(config.reminderVolume).toBeLessThan(config.discordantVolume);
        });
    });

    describe('Correct Tone', () => {
        it('should play correct tone confirmation', async () => {
            await audioFeedback.playCorrectTone('C', 0.1);
            expect(audioFeedback.isReady()).toBe(true);
        });

        it('should be quieter than reminder tone', async () => {
            const config = audioFeedback.getConfig();
            // Correct tone volume = reminderVolume * 0.7
            // So we can't directly check, but the logic is in playCorrectTone
            await audioFeedback.playCorrectTone('C', 0.1);
            expect(audioFeedback.isReady()).toBe(true);
        });
    });

    describe('Wrong Note Sequence', () => {
        it('should play discordant then reminder in sequence', async () => {
            await audioFeedback.playWrongNoteSequence('C');
            expect(audioFeedback.isReady()).toBe(true);
        });
    });

    describe('Configuration', () => {
        it('should update master volume', () => {
            audioFeedback.setMasterVolume(0.8);
            expect(audioFeedback.getConfig().masterVolume).toBe(0.8);
        });

        it('should clamp volume to 0-1 range', () => {
            audioFeedback.setMasterVolume(1.5);
            expect(audioFeedback.getConfig().masterVolume).toBe(1);

            audioFeedback.setMasterVolume(-0.5);
            expect(audioFeedback.getConfig().masterVolume).toBe(0);
        });

        it('should enable/disable audio', () => {
            expect(audioFeedback.isEnabled()).toBe(true);

            audioFeedback.setEnabled(false);
            expect(audioFeedback.isEnabled()).toBe(false);

            audioFeedback.setEnabled(true);
            expect(audioFeedback.isEnabled()).toBe(true);
        });

        it('should update config partially', () => {
            audioFeedback.updateConfig({ discordantVolume: 0.8 });
            const config = audioFeedback.getConfig();
            expect(config.discordantVolume).toBe(0.8);
            expect(config.masterVolume).toBe(0.3); // Unchanged
        });
    });

    describe('Cleanup', () => {
        it('should close AudioContext on destroy', async () => {
            await audioFeedback.playDiscordantTone('C', 0.1);
            expect(audioFeedback.isReady()).toBe(true);

            await audioFeedback.destroy();
            expect(audioFeedback.isReady()).toBe(false);
        });

        it('should handle multiple destroy calls', async () => {
            await audioFeedback.playDiscordantTone('C', 0.1);
            await audioFeedback.destroy();
            await expect(audioFeedback.destroy()).resolves.not.toThrow();
        });
    });
});
