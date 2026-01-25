/**
 * Unit tests for DuetProgressTracker
 *
 * Tests progress tracking without scores or penalties.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Duet progress calculation (0-1) - P0
 * - Sequence completion tracking - P0
 * - Full duet completion - P0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DuetProgressTracker } from '../../src/entities/DuetProgressTracker';

describe('DuetProgressTracker', () => {
    let tracker: DuetProgressTracker;

    beforeEach(() => {
        tracker = new DuetProgressTracker();
    });

    describe('Construction', () => {
        it('should initialize with zero progress', () => {
            expect(tracker.getProgress()).toBe(0);
            expect(tracker.getNotesCompleted()).toBe(0);
        });

        it('should not be complete initially', () => {
            expect(tracker.isComplete()).toBe(false);
        });

        it('should have zero attempts initially', () => {
            expect(tracker.getAttempts()).toBe(0);
        });

        it('should have total notes constant', () => {
            expect(DuetProgressTracker.getTotalNotes()).toBe(9);
        });
    });

    describe('Note Completion', () => {
        it('should increment notes completed', () => {
            tracker.markNoteComplete();
            expect(tracker.getNotesCompleted()).toBe(1);
        });

        it('should cap at total notes', () => {
            for (let i = 0; i < 15; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.getNotesCompleted()).toBe(9);
        });

        it('should calculate progress correctly', () => {
            tracker.markNoteComplete();
            expect(tracker.getProgress()).toBeCloseTo(1/9, 3);

            tracker.markNoteComplete();
            expect(tracker.getProgress()).toBeCloseTo(2/9, 3);

            tracker.markNoteComplete();
            expect(tracker.getProgress()).toBeCloseTo(3/9, 3);
        });

        it('should return 1.0 progress when complete', () => {
            for (let i = 0; i < 9; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.getProgress()).toBe(1);
        });
    });

    describe('Sequence Tracking', () => {
        it('should start in sequence 0', () => {
            expect(tracker.getCurrentSequence()).toBe(0);
        });

        it('should start at note 0 in sequence', () => {
            expect(tracker.getCurrentNote()).toBe(0);
        });

        it('should advance through sequence 0 notes', () => {
            expect(tracker.getCurrentNote()).toBe(0);

            tracker.markNoteComplete();
            expect(tracker.getCurrentNote()).toBe(1);

            tracker.markNoteComplete();
            expect(tracker.getCurrentNote()).toBe(2);

            tracker.markNoteComplete();
            expect(tracker.getCurrentNote()).toBe(0); // Reset for next sequence
        });

        it('should advance to sequence 1 after 3 notes', () => {
            for (let i = 0; i < 3; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.getCurrentSequence()).toBe(1);
            expect(tracker.getCurrentNote()).toBe(0);
        });

        it('should advance to sequence 2 after 6 notes', () => {
            for (let i = 0; i < 6; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.getCurrentSequence()).toBe(2);
            expect(tracker.getCurrentNote()).toBe(0);
        });

        it('should detect sequence completion', () => {
            expect(tracker.isSequenceComplete()).toBe(false);

            // Complete first note of sequence
            tracker.markNoteComplete();
            expect(tracker.isSequenceComplete()).toBe(false);

            // Complete rest of first sequence
            tracker.markNoteComplete();
            tracker.markNoteComplete();

            // Now at start of sequence 1, so previous sequence is complete
            expect(tracker.getCurrentNote()).toBe(0);
            expect(tracker.getNotesCompleted()).toBe(3);
        });
    });

    describe('Completion Detection', () => {
        it('should not be complete with 8 notes', () => {
            for (let i = 0; i < 8; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.isComplete()).toBe(false);
        });

        it('should be complete with 9 notes', () => {
            for (let i = 0; i < 9; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.isComplete()).toBe(true);
        });

        it('should stay complete after 9 notes', () => {
            for (let i = 0; i < 9; i++) {
                tracker.markNoteComplete();
            }
            expect(tracker.isComplete()).toBe(true);

            tracker.markNoteComplete(); // Extra
            expect(tracker.isComplete()).toBe(true);
        });
    });

    describe('Attempt Tracking', () => {
        it('should record attempts', () => {
            tracker.recordAttempt();
            expect(tracker.getAttempts()).toBe(1);

            tracker.recordAttempt();
            tracker.recordAttempt();
            expect(tracker.getAttempts()).toBe(3);
        });

        it('should not affect progress tracking', () => {
            tracker.recordAttempt();
            tracker.recordAttempt();
            expect(tracker.getNotesCompleted()).toBe(0);
            expect(tracker.getProgress()).toBe(0);
        });
    });

    describe('Reset', () => {
        it('should reset all progress', () => {
            for (let i = 0; i < 5; i++) {
                tracker.markNoteComplete();
            }
            tracker.recordAttempt();

            tracker.reset();

            expect(tracker.getNotesCompleted()).toBe(0);
            expect(tracker.getProgress()).toBe(0);
            expect(tracker.getAttempts()).toBe(0);
            expect(tracker.isComplete()).toBe(false);
        });

        it('should reset sequence and note indices', () => {
            for (let i = 0; i < 7; i++) {
                tracker.markNoteComplete();
            }

            tracker.reset();

            expect(tracker.getCurrentSequence()).toBe(0);
            expect(tracker.getCurrentNote()).toBe(0);
        });
    });

    describe('Progress Milestones', () => {
        it('should have correct progress at key milestones', () => {
            // 0/9 notes = 0%
            expect(tracker.getProgress()).toBe(0);

            // 3/9 notes = 33% (sequence 0 complete)
            for (let i = 0; i < 3; i++) tracker.markNoteComplete();
            expect(tracker.getProgress()).toBeCloseTo(0.333, 2);

            // 6/9 notes = 67% (sequence 1 complete)
            for (let i = 0; i < 3; i++) tracker.markNoteComplete();
            expect(tracker.getProgress()).toBeCloseTo(0.667, 2);

            // 9/9 notes = 100% (all complete)
            for (let i = 0; i < 3; i++) tracker.markNoteComplete();
            expect(tracker.getProgress()).toBe(1);
        });
    });

    describe('Cleanup', () => {
        it('should destroy without errors', () => {
            expect(() => tracker.destroy()).not.toThrow();
        });
    });
});
