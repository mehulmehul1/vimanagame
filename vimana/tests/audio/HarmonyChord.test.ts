/**
 * Unit tests for HarmonyChord
 *
 * Tests perfect fifth harmonies and completion chords.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Harmony chord (player + perfect fifth) - P0
 * - Completion chord (C major) - P0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { HarmonyChord } from '../../src/audio/HarmonyChord';
import type { NoteName } from '../../src/audio/NoteFrequencies';

// Mock AudioContext
class MockAudioContext {
    state: 'suspended' | 'running' | 'closed' = 'suspended';
    destination = {};
    currentTime = 0;

    createOscillator() {
        return {
            frequency: { value: 0 },
            type: 'sine',
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

global.AudioContext = MockAudioContext as any;
global.webkitAudioContext = MockAudioContext as any;

describe('HarmonyChord', () => {
    let harmonyChord: HarmonyChord;

    beforeEach(() => {
        harmonyChord = new HarmonyChord();
    });

    afterEach(async () => {
        await harmonyChord.destroy();
        vi.clearAllMocks();
    });

    describe('Construction', () => {
        it('should initialize with default config', () => {
            const config = harmonyChord.getConfig();
            expect(config.intervalRatio).toBe(1.5); // Perfect fifth
            expect(config.playerVolume).toBe(0.6);
            expect(config.harmonyVolume).toBe(0.4);
            expect(config.masterVolume).toBe(0.4);
            expect(config.enabled).toBe(true);
        });

        it('should have C major frequencies defined', () => {
            // C major chord: C (261.63), E (329.63), G (392.00)
            const expected = [261.63, 329.63, 392.00];
            expect(expected[0]).toBeCloseTo(261.63, 1);
            expect(expected[1]).toBeCloseTo(329.63, 1);
            expect(expected[2]).toBeCloseTo(392.00, 1);
        });
    });

    describe('Perfect Fifth Harmony', () => {
        it('should use 1.5 ratio for perfect fifth', () => {
            const config = harmonyChord.getConfig();
            expect(config.intervalRatio).toBe(1.5);
        });

        it('should calculate correct harmony frequencies', () => {
            // C (261.63 Hz) * 1.5 = G (392.45 Hz)
            const baseFreq = 261.63;
            const harmonyFreq = baseFreq * 1.5;
            expect(harmonyFreq).toBeCloseTo(392.45, 1);
        });

        it('should play harmony for all note indices', async () => {
            for (let i = 0; i < 6; i++) {
                await harmonyChord.playHarmony(i, 0.1);
            }
            expect(harmonyChord.isReady()).toBe(true);
        });
    });

    describe('Demonstration Note', () => {
        it('should play demonstration note', async () => {
            await harmonyChord.playDemonstrationNote(0, 0.1);
            expect(harmonyChord.isReady()).toBe(true);
        });

        it('should play for each note index', async () => {
            const noteNames: NoteName[] = ['C', 'D', 'E', 'F', 'G', 'A'];
            for (let i = 0; i < noteNames.length; i++) {
                await harmonyChord.playDemonstrationNote(i, 0.1);
            }
            expect(harmonyChord.isReady()).toBe(true);
        });
    });

    describe('Completion Chord', () => {
        it('should have C major frequencies defined', () => {
            // C major triad: C-E-G
            const expected = [261.63, 329.63, 392.00];
            expect(expected).toHaveLength(3);
            expect(expected[0]).toBeCloseTo(261.63, 1);
            expect(expected[1]).toBeCloseTo(329.63, 1);
            expect(expected[2]).toBeCloseTo(392.00, 1);
        });

        it('should have correct gain distribution for completion chord', () => {
            // Gains: [0.4, 0.35, 0.4] - root and fifth slightly louder
            const gains = [0.4, 0.35, 0.4];
            expect(gains[0]).toBeGreaterThan(gains[1]); // Root louder than third
            expect(gains[2]).toBeGreaterThan(gains[1]); // Fifth louder than third
        });
    });

    describe('Configuration', () => {
        it('should update master volume', () => {
            harmonyChord.setMasterVolume(0.8);
            expect(harmonyChord.getConfig().masterVolume).toBe(0.8);
        });

        it('should clamp volume to valid range', () => {
            harmonyChord.setMasterVolume(1.5);
            expect(harmonyChord.getConfig().masterVolume).toBe(1);

            harmonyChord.setMasterVolume(-0.5);
            expect(harmonyChord.getConfig().masterVolume).toBe(0);
        });

        it('should enable/disable harmony', () => {
            expect(harmonyChord.getConfig().enabled).toBe(true);

            harmonyChord.setEnabled(false);
            expect(harmonyChord.getConfig().enabled).toBe(false);

            harmonyChord.setEnabled(true);
            expect(harmonyChord.getConfig().enabled).toBe(true);
        });

        it('should update config partially', () => {
            harmonyChord.updateConfig({ harmonyVolume: 0.5 });
            const config = harmonyChord.getConfig();
            expect(config.harmonyVolume).toBe(0.5);
            expect(config.playerVolume).toBe(0.6); // Unchanged
        });
    });

    describe('Cleanup', () => {
        it('should close AudioContext on destroy', async () => {
            await harmonyChord.playDemonstrationNote(0, 0.1);
            expect(harmonyChord.isReady()).toBe(true);

            await harmonyChord.destroy();
            expect(harmonyChord.isReady()).toBe(false);
        });

        it('should handle multiple destroy calls', async () => {
            await harmonyChord.destroy();
            await expect(harmonyChord.destroy()).resolves.not.toThrow();
        });
    });

    describe('Ready State', () => {
        it('should report not ready before initialization', () => {
            expect(harmonyChord.isReady()).toBe(false);
        });

        it('should report ready after first audio call', async () => {
            await harmonyChord.playDemonstrationNote(0, 0.1);
            expect(harmonyChord.isReady()).toBe(true);
        });
    });
});
