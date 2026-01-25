/**
 * NoteFrequencies - Musical note frequencies for the harp
 *
 * Standard frequencies for the 6 harp strings (C4-A4).
 * Used for audio feedback and string identification.
 */

export const NOTE_FREQUENCIES = {
    /** Middle C */
    C: 261.63,
    /** D above middle C */
    D: 293.66,
    /** E above middle C */
    E: 329.63,
    /** F above middle C */
    F: 349.23,
    /** G above middle C */
    G: 392.00,
    /** A above middle C */
    A: 440.00
} as const;

export type NoteName = keyof typeof NOTE_FREQUENCIES;

/**
 * Get frequency for note index (0-5 for C-A)
 */
export function getFrequencyByIndex(index: number): number {
    const noteNames: NoteName[] = ['C', 'D', 'E', 'F', 'G', 'A'];
    if (index >= 0 && index < noteNames.length) {
        return NOTE_FREQUENCIES[noteNames[index]];
    }
    return NOTE_FREQUENCIES.C; // Default to C
}

/**
 * Calculate discordant frequency using minor second interval
 * @param baseFreq The correct frequency
 * @param up If true, go up a semitone; if false, go down
 */
export function getDiscordantFrequency(baseFreq: number, up: boolean = true): number {
    if (up) {
        // Minor second up (semitone): 16/15 ratio
        return baseFreq * (16 / 15);
    } else {
        // Minor second down: 15/16 ratio
        return baseFreq * (15 / 16);
    }
}

/**
 * Note names array for indexing
 */
export const NOTE_NAMES: NoteName[] = ['C', 'D', 'E', 'F', 'G', 'A'];

/**
 * Reminder tone frequency (gentle E4)
 */
export const REMINDER_FREQUENCY = 329.63; // E4
