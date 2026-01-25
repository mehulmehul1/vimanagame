/**
 * PatientJellyManager - Teaching system for duet mechanics
 *
 * Manages jelly creature demonstrations and player input for
 * learning the harp sequences. No failure state—only patient teaching.
 *
 * Philosophy: "The ship teaches you, it doesn't test you."
 */

import { JellyManager } from './JellyManager';
import { HarmonyChord } from '../audio/HarmonyChord';
import { DuetProgressTracker } from './DuetProgressTracker';
import { FeedbackManager } from './FeedbackManager';

/**
 * Duet teaching states
 */
export enum DuetState {
    /** Waiting to start or between sequences */
    IDLE = 'IDLE',
    /** Jelly is demonstrating which note to play */
    DEMONSTRATING = 'DEMONSTRATING',
    /** Jelly submerged, waiting for player input */
    AWAITING_INPUT = 'AWAITING_INPUT',
    /** Playing harmony chord after correct note */
    PLAYING_HARMONY = 'PLAYING_HARMONY',
    /** All sequences complete */
    COMPLETE = 'COMPLETE'
}

/**
 * Teaching sequences
 * Each array contains string indices (0-5) for notes to teach
 */
export const TEACHING_SEQUENCES = [
    [0, 1, 2], // Sequence 1: C, D, E (lower register)
    [3, 4, 5], // Sequence 2: F, G, A (upper register)
    [2, 4, 1]  // Sequence 3: E, G, D (melodic phrase)
] as const;

export interface DuetCallbacks {
    /** Called when a note is completed correctly */
    onNoteComplete?: (sequenceIndex: number, noteIndex: number) => void;
    /** Called when a sequence is completed */
    onSequenceComplete?: (sequenceIndex: number) => void;
    /** Called when entire duet is complete */
    onDuetComplete?: () => void;
    /** Called when wrong note is played */
    onWrongNote?: (targetNoteIndex: number, playedNoteIndex: number) => void;
    /** Called when demonstration starts (jelly spawns) */
    onDemonstrationStart?: (noteIndex: number) => void;
    /** Called when demonstration ends (jelly submerges) */
    onDemonstrationEnd?: (noteIndex: number) => void;
}

export interface PatientJellyConfig {
    /** Demonstration duration in seconds */
    demoDuration: number;
    /** Re-demonstration duration (slower) in seconds */
    redemoDuration: number;
    /** Delay before re-demonstration after wrong note in seconds */
    wrongNoteDelay: number;
    /** Delay between sequences in seconds */
    sequenceDelay: number;
}

export class PatientJellyManager {
    private jellyManager: JellyManager;
    private harmonyChord: HarmonyChord;
    private feedbackManager: FeedbackManager;
    private progressTracker: DuetProgressTracker;

    private state: DuetState = DuetState.IDLE;
    private currentSequence: number = 0;
    private currentNoteIndex: number = 0;
    private isRedemonstrating: boolean = false;

    private config: PatientJellyConfig;
    private callbacks: DuetCallbacks;

    // Timeout IDs for cleanup
    private demoTimeout: number | null = null;
    private redemoTimeout: number | null = null;
    private sequenceTimeout: number | null = null;

    constructor(
        jellyManager: JellyManager,
        harmonyChord: HarmonyChord,
        feedbackManager: FeedbackManager,
        callbacks: DuetCallbacks = {},
        config: Partial<PatientJellyConfig> = {}
    ) {
        this.jellyManager = jellyManager;
        this.harmonyChord = harmonyChord;
        this.feedbackManager = feedbackManager;
        this.progressTracker = new DuetProgressTracker();

        this.callbacks = callbacks;
        this.config = {
            demoDuration: 2.0,
            redemoDuration: 3.0,
            wrongNoteDelay: 1.0,
            sequenceDelay: 2.0,
            ...config
        };
    }

    /**
     * Start teaching a specific sequence
     */
    public startSequence(sequenceIndex: number): void {
        if (this.state === DuetState.COMPLETE) return;
        if (sequenceIndex < 0 || sequenceIndex >= TEACHING_SEQUENCES.length) return;

        this.currentSequence = sequenceIndex;
        this.currentNoteIndex = 0;
        this.isRedemonstrating = false;

        this.demonstrateCurrentNote();
    }

    /**
     * Start the duet from the beginning
     */
    public start(): void {
        this.startSequence(0);
    }

    /**
     * Demonstrate the current note
     */
    private demonstrateCurrentNote(): void {
        this.state = DuetState.DEMONSTRATING;

        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        // Notify callback that demonstration is starting
        if (this.callbacks.onDemonstrationStart) {
            this.callbacks.onDemonstrationStart(targetNote);
        }

        // Spawn and teach with jelly
        this.jellyManager.spawnJelly(targetNote);
        setTimeout(() => {
            this.jellyManager.beginTeaching();
        }, 100);

        // Play demonstration note
        this.harmonyChord.playDemonstrationNote(targetNote, this.isRedemonstrating ?
            this.config.redemoDuration : this.config.demoDuration);

        // Schedule transition to awaiting input
        const demoDuration = this.isRedemonstrating ?
            this.config.redemoDuration : this.config.demoDuration;

        this.demoTimeout = window.setTimeout(() => {
            this.transitionToAwaitingInput();
        }, demoDuration * 1000);
    }

    /**
     * Transition to awaiting input state
     */
    private transitionToAwaitingInput(): void {
        this.state = DuetState.AWAITING_INPUT;

        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        // Notify callback that demonstration is ending
        if (this.callbacks.onDemonstrationEnd) {
            this.callbacks.onDemonstrationEnd(targetNote);
        }

        this.jellyManager.submergeActive();
    }

    /**
     * Handle player input (harp string played)
     */
    public handlePlayerInput(playedNoteIndex: number): void {
        // Log for debugging
        console.log(`[PatientJellyManager] handlePlayerInput: note=${playedNoteIndex}, state=${this.state}`);

        // Allow input during DEMONSTRATING (player can play along!)
        // Only ignore if COMPLETE
        if (this.state === DuetState.COMPLETE) {
            console.log('[PatientJellyManager] Ignoring input - duet complete');
            return;
        }

        // If playing during demonstration, skip to AWAITING_INPUT immediately
        if (this.state === DuetState.DEMONSTRATING) {
            console.log('[PatientJellyManager] Player played during demo, skipping to input phase');
            // Clear the demo timeout
            if (this.demoTimeout) {
                clearTimeout(this.demoTimeout);
                this.demoTimeout = null;
            }
            this.transitionToAwaitingInput();
        }

        this.progressTracker.recordAttempt();

        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        if (playedNoteIndex === targetNote) {
            this.handleCorrectNote();
        } else {
            this.handleWrongNote(targetNote, playedNoteIndex);
        }
    }

    /**
     * Handle correct note played
     */
    private handleCorrectNote(): void {
        this.state = DuetState.PLAYING_HARMONY;
        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        // Play harmony chord
        this.harmonyChord.playHarmony(targetNote);

        // Mark progress
        this.progressTracker.markNoteComplete();

        // Trigger visual feedback
        this.feedbackManager.triggerCorrectNote(targetNote);

        // Callback
        if (this.callbacks.onNoteComplete) {
            this.callbacks.onNoteComplete(this.currentSequence, this.currentNoteIndex);
        }

        // Clear any pending redemo
        if (this.redemoTimeout) {
            clearTimeout(this.redemoTimeout);
            this.redemoTimeout = null;
        }

        // Advance after harmony plays
        this.demoTimeout = window.setTimeout(() => {
            this.advanceToNextNote();
        }, 600);
    }

    /**
     * Handle wrong note played
     */
    private handleWrongNote(targetNote: number, playedNote: number): void {
        // Trigger gentle feedback (shake + discordant + highlight)
        this.feedbackManager.triggerWrongNote(targetNote);

        // Callback
        if (this.callbacks.onWrongNote) {
            this.callbacks.onWrongNote(targetNote, playedNote);
        }

        // Stay in AWAITING_INPUT, schedule re-demonstration
        // Same note, no progress lost
        this.redemoTimeout = window.setTimeout(() => {
            this.isRedemonstrating = true;
            this.demonstrateCurrentNote();
        }, this.config.wrongNoteDelay * 1000);
    }

    /**
     * Advance to next note or complete sequence
     */
    private advanceToNextNote(): void {
        this.currentNoteIndex++;
        this.isRedemonstrating = false;

        // Check if we've completed the current sequence
        const currentSequence = TEACHING_SEQUENCES[this.currentSequence];
        if (this.currentNoteIndex >= currentSequence.length) {
            this.completeSequence();
        } else {
            this.demonstrateCurrentNote();
        }
    }

    /**
     * Complete current sequence
     */
    private completeSequence(): void {
        // Play completion chord
        this.harmonyChord.playCompletionChord();

        // Callback
        if (this.callbacks.onSequenceComplete) {
            this.callbacks.onSequenceComplete(this.currentSequence);
        }

        // Check if all sequences complete
        if (this.currentSequence >= TEACHING_SEQUENCES.length - 1) {
            this.completeDuet();
        } else {
            // Move to next sequence after delay
            this.sequenceTimeout = window.setTimeout(() => {
                this.currentSequence++;
                this.currentNoteIndex = 0;
                this.startSequence(this.currentSequence);
            }, this.config.sequenceDelay * 1000);
        }
    }

    /**
     * Complete entire duet
     */
    private completeDuet(): void {
        this.state = DuetState.COMPLETE;

        if (this.callbacks.onDuetComplete) {
            this.callbacks.onDuetComplete();
        }
    }

    /**
     * Get current state
     */
    public getState(): DuetState {
        return this.state;
    }

    /**
     * Get current sequence index
     */
    public getCurrentSequence(): number {
        return this.currentSequence;
    }

    /**
     * Get current note index within sequence
     */
    public getCurrentNoteIndex(): number {
        return this.currentNoteIndex;
    }

    /**
     * Get target note for current step
     */
    public getTargetNote(): number {
        if (this.state === DuetState.COMPLETE) return -1;
        return TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];
    }

    /**
     * Get overall progress (0-1)
     */
    public getProgress(): number {
        return this.progressTracker.getProgress();
    }

    /**
     * Check if duet is complete
     */
    public isComplete(): boolean {
        return this.state === DuetState.COMPLETE;
    }

    /**
     * Check if currently awaiting input
     */
    public isAwaitingInput(): boolean {
        return this.state === DuetState.AWAITING_INPUT;
    }

    /**
     * Get teaching sequences
     */
    public static getSequences(): readonly (readonly number[])[] {
        return TEACHING_SEQUENCES;
    }

    /**
     * Update callbacks
     */
    public setCallbacks(callbacks: Partial<DuetCallbacks>): void {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<PatientJellyConfig>): void {
        this.config = { ...this.config, ...updates };
    }

    /**
     * Pause teaching (stop all timeouts)
     */
    public pause(): void {
        if (this.demoTimeout) {
            clearTimeout(this.demoTimeout);
            this.demoTimeout = null;
        }
        if (this.redemoTimeout) {
            clearTimeout(this.redemoTimeout);
            this.redemoTimeout = null;
        }
        if (this.sequenceTimeout) {
            clearTimeout(this.sequenceTimeout);
            this.sequenceTimeout = null;
        }
    }

    /**
     * Resume teaching
     */
    public resume(): void {
        if (this.state === DuetState.DEMONSTRATING) {
            this.demonstrateCurrentNote();
        }
    }

    /**
     * Reset and start over
     */
    public reset(): void {
        this.pause();
        this.state = DuetState.IDLE;
        this.currentSequence = 0;
        this.currentNoteIndex = 0;
        this.isRedemonstrating = false;
        this.progressTracker.reset();
        this.jellyManager.submergeActive();
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.pause();
        this.state = DuetState.IDLE;
        this.progressTracker.destroy();
    }
}
