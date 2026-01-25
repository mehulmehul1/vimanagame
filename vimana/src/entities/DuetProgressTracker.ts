/**
 * DuetProgressTracker - Tracks duet completion progress
 *
 * Monitors player progress through the teaching sequences.
 * No scores, no penalties—only forward progress.
 */

export class DuetProgressTracker {
    /** Total number of notes across all sequences */
    private static readonly TOTAL_NOTES = 9; // 3 sequences × 3 notes

    /** Number of notes successfully completed */
    private notesCompleted: number = 0;

    /** Number of attempts made (for analytics only, never shown to player) */
    private attempts: number = 0;

    /** Whether duet is fully complete */
    private isFullyComplete: boolean = false;

    /**
     * Mark a note as completed
     * Called when player successfully plays the target note
     */
    public markNoteComplete(): void {
        this.notesCompleted = Math.min(this.notesCompleted + 1, DuetProgressTracker.TOTAL_NOTES);

        if (this.notesCompleted >= DuetProgressTracker.TOTAL_NOTES) {
            this.isFullyComplete = true;
        }
    }

    /**
     * Record an attempt (wrong or right)
     * For analytics only—never affects gameplay
     */
    public recordAttempt(): void {
        this.attempts++;
    }

    /**
     * Get overall progress (0-1)
     */
    public getProgress(): number {
        return this.notesCompleted / DuetProgressTracker.TOTAL_NOTES;
    }

    /**
     * Get current sequence index (0-2)
     */
    public getCurrentSequence(): number {
        return Math.floor(this.notesCompleted / 3);
    }

    /**
     * Get current note index within sequence (0-2)
     */
    public getCurrentNote(): number {
        return this.notesCompleted % 3;
    }

    /**
     * Get total notes completed
     */
    public getNotesCompleted(): number {
        return this.notesCompleted;
    }

    /**
     * Check if duet is complete
     */
    public isComplete(): boolean {
        return this.isFullyComplete;
    }

    /**
     * Get number of attempts (for analytics)
     */
    public getAttempts(): number {
        return this.attempts;
    }

    /**
     * Check if current sequence is complete
     */
    public isSequenceComplete(): boolean {
        return this.getCurrentNote() === 0 && this.notesCompleted > 0;
    }

    /**
     * Reset progress (for testing/new game)
     * Normally not called—duet progress is permanent
     */
    public reset(): void {
        this.notesCompleted = 0;
        this.attempts = 0;
        this.isFullyComplete = false;
    }

    /**
     * Get total notes in duet
     */
    public static getTotalNotes(): number {
        return DuetProgressTracker.TOTAL_NOTES;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        // No resources to clean up
    }
}
