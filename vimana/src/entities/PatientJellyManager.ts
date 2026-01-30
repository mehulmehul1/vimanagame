/**
 * PatientJellyManager - Teaching system for duet mechanics
 *
 * Manages jelly creature demonstrations and player input for
 * learning the harp sequences. No failure state—only patient teaching.
 *
 * Philosophy: "The ship teaches you, it doesn't test you."
 */

import { JellyManager } from './JellyManager';
import { JellyCreature } from './JellyCreature';
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
    COMPLETE = 'COMPLETE',
    /** All jellies demonstrating full phrase (STORY-HARP-101) */
    PHRASE_DEMONSTRATION = 'PHRASE_DEMONSTRATION',
    /** Synchronized splash turn signal (STORY-HARP-101) */
    TURN_SIGNAL = 'TURN_SIGNAL',
    /** Player's turn to replay phrase (STORY-HARP-101) */
    AWAITING_PHRASE_RESPONSE = 'AWAITING_PHRASE_RESPONSE'
}

/**
 * Teaching mode configuration (STORY-HARP-101)
 */
export interface TeachingModeConfig {
    /** Active teaching mode */
    mode: 'phrase-first' | 'note-by-note';
    /** Show all jellies at once (phrase-first) */
    showAllJellies: boolean;
    /** Use synchronized splash turn signal */
    synchronizedSplash: boolean;
    /** Delay between jelly emergence in milliseconds */
    jellyStaggerMs: number;
    /** Duration each demonstration note plays */
    demoNoteDuration: number;
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
    /** Called when a note is demonstrated in phrase-first mode (STORY-HARP-101) */
    onNoteDemonstrated?: (noteIndex: number, sequenceIndex: number) => void;
    /** Called when synchronized splash completes (STORY-HARP-102) */
    onTurnSignalComplete?: () => void;
    /** Called when synchronized splash triggers (STORY-HARP-102) */
    onSynchronizedSplash?: (positions: number[]) => void;
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

    // STORY-HARP-101: Teaching mode support
    private teachingMode: 'phrase-first' | 'note-by-note' = 'phrase-first';
    private teachingModeConfig: TeachingModeConfig = {
        mode: 'phrase-first',
        showAllJellies: true,
        synchronizedSplash: true,
        jellyStaggerMs: 800,
        demoNoteDuration: 1.5
    };
    // Track active jellies for phrase-first mode
    private activeJellies: Map<number, JellyCreature> = new Map();
    private phraseTimeouts: number[] = [];

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

        if (this.teachingMode === 'phrase-first') {
            this.startPhraseFirstSequence(sequenceIndex);
        } else {
            this.demonstrateCurrentNote();
        }
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
     * Works for both phrase-first and note-by-note modes
     */
    public handlePlayerInput(playedNoteIndex: number): void {
        // Log for debugging
        console.log(`[PatientJellyManager] handlePlayerInput: note=${playedNoteIndex}, state=${this.state}`);

        // Ignore if complete
        if (this.state === DuetState.COMPLETE) {
            console.log('[PatientJellyManager] Ignoring input - duet complete');
            return;
        }

        // PHRASE-FIRST MODE: Validate against current expected note in sequence
        if (this.state === DuetState.AWAITING_PHRASE_RESPONSE) {
            this.handlePhraseFirstInput(playedNoteIndex);
            return;
        }

        // NOTE-BY-NOTE MODE: Existing logic
        if (this.state === DuetState.AWAITING_INPUT) {
            this.handleNoteByNoteInput(playedNoteIndex);
            return;
        }

        // Allow playing during PHRASE_DEMONSTRATION (skip to player's turn)
        if (this.state === DuetState.PHRASE_DEMONSTRATION) {
            console.log('[PatientJellyManager] Player played during phrase demo, ending demo');
            this.endDemonstrationEarly();
            this.state = DuetState.AWAITING_PHRASE_RESPONSE;
            this.currentNoteIndex = 0;
            this.handlePhraseFirstInput(playedNoteIndex);
            return;
        }

        // Allow playing during single note demonstration
        if (this.state === DuetState.DEMONSTRATING) {
            console.log('[PatientJellyManager] Player played during demo, skipping to input phase');
            if (this.demoTimeout) {
                clearTimeout(this.demoTimeout);
                this.demoTimeout = null;
            }
            this.transitionToAwaitingInput();
            this.handleNoteByNoteInput(playedNoteIndex);
        }
    }

    /**
     * Handle input during phrase-first response phase
     */
    private handlePhraseFirstInput(playedNoteIndex: number): void {
        this.progressTracker.recordAttempt();

        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        console.log(`[Phrase-First] Expecting note ${targetNote}, played ${playedNoteIndex} (${this.currentNoteIndex + 1}/${TEACHING_SEQUENCES[this.currentSequence].length})`);

        if (playedNoteIndex === targetNote) {
            this.handleCorrectNoteInPhrase();
        } else {
            this.handleWrongNoteInPhrase(targetNote, playedNoteIndex);
        }
    }

    /**
     * Handle input during note-by-note phase (legacy logic)
     */
    private handleNoteByNoteInput(playedNoteIndex: number): void {
        this.progressTracker.recordAttempt();

        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        if (playedNoteIndex === targetNote) {
            this.handleCorrectNote();
        } else {
            this.handleWrongNote(targetNote, playedNoteIndex);
        }
    }

    /**
     * Handle correct note within phrase response
     */
    private handleCorrectNoteInPhrase(): void {
        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        // Play individual note confirmation (subtle feedback)
        this.harmonyChord.playNoteConfirmation(targetNote);

        // Trigger visual feedback
        this.feedbackManager.triggerCorrectNote(targetNote);

        // Callback
        if (this.callbacks.onNoteComplete) {
            this.callbacks.onNoteComplete(this.currentSequence, this.currentNoteIndex);
        }

        // Advance to next note in phrase
        this.currentNoteIndex++;
        const currentSequence = TEACHING_SEQUENCES[this.currentSequence];

        if (this.currentNoteIndex >= currentSequence.length) {
            // Whole phrase completed correctly!
            this.completePhrase();
        } else {
            console.log(`[Phrase-First] Correct! ${this.currentNoteIndex}/${currentSequence.length} complete. Next note expected.`);
        }
    }

    /**
     * Handle wrong note during phrase response
     */
    private handleWrongNoteInPhrase(targetNote: number, playedNote: number): void {
        console.log(`[Phrase-First] Wrong note! Expected ${targetNote}, played ${playedNote}`);

        // Gentle feedback (shake + discordant + highlight)
        this.feedbackManager.triggerWrongNote(targetNote);

        // Callback
        if (this.callbacks.onWrongNote) {
            this.callbacks.onWrongNote(targetNote, playedNote);
        }

        // IMPORTANT: Patient teaching - redemonstrate the full phrase
        // No punishment, no progress lost - gentle retry
        setTimeout(() => {
            console.log('[Phrase-First] Redemonstrating phrase...');
            this.startPhraseFirstSequence(this.currentSequence);
        }, this.config.wrongNoteDelay * 1000);
    }

    /**
     * Phrase completed successfully
     */
    private completePhrase(): void {
        console.log(`[Phrase-First] Sequence ${this.currentSequence} complete!`);

        // Play full harmony chord (ship joins in)
        this.harmonyChord.playCompletionChord();

        // Mark progress
        this.progressTracker.markSequenceComplete(this.currentSequence);

        // Trigger visual celebration
        this.feedbackManager.triggerSequenceComplete(this.currentSequence);

        // Callback
        if (this.callbacks.onSequenceComplete) {
            this.callbacks.onSequenceComplete(this.currentSequence);
        }

        // Move to next sequence or complete duet
        if (this.currentSequence >= TEACHING_SEQUENCES.length - 1) {
            this.completeDuet();
        } else {
            // Delay before starting next sequence
            setTimeout(() => {
                this.currentSequence++;
                this.startSequence(this.currentSequence);
            }, this.config.sequenceDelay * 1000);
        }
    }

    /**
     * Helper to end a phrase demonstration early
     */
    private endDemonstrationEarly(): void {
        // Clear all phrase timeouts
        this.phraseTimeouts.forEach(id => clearTimeout(id));
        this.phraseTimeouts = [];

        // Stop all active jellies
        this.jellyManager.submergeActive();
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
        // STORY-HARP-101: Clear phrase timeouts
        for (const timeoutId of this.phraseTimeouts) {
            clearTimeout(timeoutId);
        }
        this.phraseTimeouts = [];
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
        // STORY-HARP-101: Clear active jellies
        this.activeJellies.clear();
        this.jellyManager.submergeActive();
    }

    // ============================================================
    // STORY-HARP-101: Phrase-First Teaching Mode
    // ============================================================

    /**
     * Set teaching mode (phrase-first or note-by-note)
     */
    public setTeachingMode(mode: 'phrase-first' | 'note-by-note'): void {
        this.teachingMode = mode;
        this.teachingModeConfig.mode = mode;
        this.teachingModeConfig.showAllJellies = (mode === 'phrase-first');
        this.teachingModeConfig.synchronizedSplash = (mode === 'phrase-first');
        console.log(`[PatientJellyManager] Teaching mode set to: ${mode}`);
    }

    /**
     * Get current teaching mode
     */
    public getTeachingMode(): 'phrase-first' | 'note-by-note' {
        return this.teachingMode;
    }

    /**
     * Start phrase-first demonstration (all jellies show sequence)
     * All jellyfish emerge sequentially to demonstrate the full phrase,
     * then wait for the player to remember and replay the entire sequence.
     */
    public startPhraseFirstSequence(sequenceIndex: number): void {
        if (this.state === DuetState.COMPLETE) return;
        if (sequenceIndex < 0 || sequenceIndex >= TEACHING_SEQUENCES.length) return;

        console.log(`[PatientJellyManager] Starting phrase-first sequence ${sequenceIndex}`);

        this.state = DuetState.PHRASE_DEMONSTRATION;
        this.currentSequence = sequenceIndex;
        this.currentNoteIndex = 0;
        this.isRedemonstrating = false;

        // Clear any previous active jellies
        this.activeJellies.clear();

        const sequence = TEACHING_SEQUENCES[sequenceIndex];
        const staggerMs = this.teachingModeConfig.jellyStaggerMs;

        // Spawn ALL jellies for the sequence with staggered timing
        for (let i = 0; i < sequence.length; i++) {
            const noteIndex = sequence[i];

            // Schedule each jelly's emergence
            const timeoutId = window.setTimeout(() => {
                // Spawn jelly at target string position
                this.jellyManager.spawnJelly(noteIndex);
                const jelly = this.jellyManager.getJelly(noteIndex);

                if (jelly) {
                    this.activeJellies.set(noteIndex, jelly);

                    // Start teaching animation after brief delay
                    setTimeout(() => {
                        this.jellyManager.beginTeaching();
                        this.harmonyChord.playDemonstrationNote(noteIndex, this.teachingModeConfig.demoNoteDuration);
                    }, 100);

                    // Trigger callback for visual indicators
                    if (this.callbacks.onNoteDemonstrated) {
                        this.callbacks.onNoteDemonstrated(noteIndex, i);
                    }

                    console.log(`[PatientJellyManager] Demonstrated note ${noteIndex} (#${i + 1} in phrase)`);
                }
            }, i * staggerMs);

            this.phraseTimeouts.push(timeoutId);
        }

        // After all notes demonstrated, trigger synchronized splash (turn signal)
        const totalDemoTime = sequence.length * staggerMs + (this.teachingModeConfig.demoNoteDuration * 1000) + 500;
        const splashTimeoutId = window.setTimeout(() => {
            this.triggerSynchronizedSplash(sequence);
        }, totalDemoTime);

        this.phraseTimeouts.push(splashTimeoutId);
    }

    /**
     * Trigger synchronized splash (STORY-HARP-102)
     * All jellies submerge together as the "your turn" signal
     *
     * This is the TURN SIGNAL that clearly communicates:
     * "The Vimana's phrase is complete. Your turn."
     */
    private triggerSynchronizedSplash(sequence: readonly number[]): void {
        console.log('[PatientJellyManager] Triggering synchronized splash (turn signal)');
        this.state = DuetState.TURN_SIGNAL;

        // STORY-HARP-102: Submerge all active jellies simultaneously
        // Use JellyManager.submergeAll() for synchronized descent
        this.jellyManager.submergeAll();

        // Clear active jellies tracking
        this.activeJellies.clear();

        // STORY-HARP-102: Play unified splash sound (single sound, not multiple)
        this.harmonyChord.playSplashSound();

        // Trigger visual splash effect callback
        if (this.callbacks.onSynchronizedSplash) {
            this.callbacks.onSynchronizedSplash([...sequence]);
        }

        // Transition to awaiting player response after splash animation
        const SPLASH_DURATION = 1000; // 1 second for splash animation
        const splashTimeoutId = window.setTimeout(() => {
            this.transitionToAwaitingPhraseResponse();
        }, SPLASH_DURATION);

        this.phraseTimeouts.push(splashTimeoutId);
    }

    /**
     * Transition to awaiting phrase response state
     */
    private transitionToAwaitingPhraseResponse(): void {
        this.state = DuetState.AWAITING_PHRASE_RESPONSE;
        this.currentNoteIndex = 0; // Player starts from beginning of phrase
        console.log('[PatientJellyManager] Player\'s turn - awaiting phrase response');

        // Trigger callback for turn signal complete
        if (this.callbacks.onTurnSignalComplete) {
            this.callbacks.onTurnSignalComplete();
        }
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
