# Story 1.4: Duet Mechanics - Patient Teaching

Status: ready-for-dev

<!-- Note: Validation is optional. run validate-create-story for quality check before dev-story. -->

## Story

As a **player learning to harmonize with the Vimana**,
I want **jelly creatures to teach me note sequences through patient demonstration and response**,
so that **I experience a duet—a shared moment of music-making, not a test to pass**.

## Acceptance Criteria

1. [ ] PatientJellyManager class with three teaching sequences
2. [ ] Sequence 1: C, D, E (strings 0, 1, 2) - introduction to the lower register
3. [ ] Sequence 2: F, G, A (strings 3, 4, 5) - introduction to the upper register
4. [ ] Sequence 3: E, G, D (strings 2, 4, 1) - a melodic phrase
5. [ ] Teaching flow: Jelly demonstrates → Player plays → Harmony chord OR gentle correction
6. [ ] No failure state—jelly simply reappears to demonstrate again
7. [ ] Harmony chord plays on correct note (player frequency × 1.5 = perfect fifth)
8. [ ] Completion chord (C major: C+E+G) on sequence finish

## Tasks / Subtasks

- [ ] Create PatientJellyManager state machine (AC: #1)
  - [ ] States: IDLE, DEMONSTRATING, AWAITING_INPUT, PLAYING_HARMONY, COMPLETE
  - [ ] Current sequence tracking (0, 1, or 2)
  - [ ] Current note index within sequence (0-2 for each sequence)
  - [ ] Teaching sequences defined as constant arrays
  - [ ] Transition methods: startSequence(), demonstrateNote(), awaitInput(), handleInput(), completeNote()
  - [ ] State persistence—no reset on wrong note
- [ ] Define teaching sequences (AC: #2, #3, #4)
  - [ ] **Sequence 1** (Introductory): [0, 1, 2] = [C, D, E]
    - Narrative: "The ship teaches you its lowest voice—warm and grounding"
  - [ ] **Sequence 2** (Ascending): [3, 4, 5] = [F, G, A]
    - Narrative: "The ship reveals its higher voice—bright and hopeful"
  - [ ] **Sequence 3** (Melodic): [2, 4, 1] = [E, G, D]
    - Narrative: "A phrase that ascends and resolves—the ship speaks to you"
  - [ ] Each sequence has 3 notes, total 9 notes to learn
- [ ] Implement demonstration phase (AC: #5)
  - [ ] Trigger JellyManager.demonstrateNote(noteIndex)
  - [ ] Jelly emerges, lands on string, teaches for 2 seconds
  - [ ] During demonstration: play the target note through speakers
  - [ ] Visual: jelly pulses at teaching rate, string glows
  - [ ] Audio: pure sine wave of target frequency
  - [ ] After demo: state transitions to AWAITING_INPUT
  - [ ] Jelly submerges but remains "present" conceptually
- [ ] Implement player input handling (AC: #5, #6)
  - [ ] Listen for harp string interactions (raycast or click)
  - [ ] On string play: check against target note
  - [ ] **Correct note:**
    - Play harmony chord (player + perfect fifth)
    - Visual: water ripples, bioluminescence intensifies
    - Audio: rich chord with both frequencies
    - Advance to next note or complete sequence
  - [ ] **Wrong note:**
    - Trigger GentleFeedback (from Story 1.3)
    - State: AWAITING_INPUT (no reset)
    - Same jelly re-emerges after 1 second to demonstrate again
    - Patient re-teaching—no counter, no penalty
- [ ] Create harmony chord system (AC: #7)
  - [ ] Perfect fifth calculation: `harmonyFreq = playerFreq × 1.5`
  - [ ] Play both frequencies simultaneously on correct input
  - [ ] Use two oscillators with slight phase offset for richness
  - [ ] Gain envelope: attack 50ms, sustain 200ms, release 300ms
  - [ ] Duration: 550ms total per harmony chord
  - [ ] Master mix: player 60%, harmony 40% (harmony supports, doesn't overpower)
- [ ] Implement sequence completion (AC: #8)
  - [ ] After all 3 notes in sequence played correctly
  - [ ] Play C major completion chord: C (261.63 Hz) + E (329.63 Hz) + G (392.00 Hz)
  - [ ] Extended duration: 2 seconds for triumph
  - [ ] Visual: vortex activation increases (see Story 1.5)
  - [ ] Water surface: full harmonic resonance glow
  - [ ] All 6 strings briefly glow together
  - [ ] State transitions to next sequence or COMPLETE if all done
- [ ] Create DuetProgressTracker (AC: #5, #8)
  - [ ] Tracks notes completed per sequence
  - [ ] Calculates overall duet progress (0-1)
  - [ ] Exposes getProgress() for vortex activation
  - [ ] Exposes getCurrentSequence() and getCurrentNote()
  - [ ] Stores attempts count (for analytics only, never shown to player)
  - [ ] No "score"—only completion matters
- [ ] Implement patient re-teaching behavior (AC: #6)
  - [ ] On wrong note: wait 500ms, then trigger jelly re-emergence
  - [ ] Jelly demonstrates same note again (no advancement)
  - [ ] Demonstration is slower on repeat (3 seconds instead of 2)
  - [ ] Jelly glows warmer (more amber) to indicate "try again"
  - [ ] No limit on demonstrations—jelly teaches forever if needed
  - [ ] State remains at current note index (no progress lost)
- [ ] Connect to existing systems (AC: #5)
  - [ ] JellyManager (from Story 1.2) for visual demonstrations
  - [ ] GentleFeedback (from Story 1.3) for wrong notes
  - [ ] HarpString entities for input detection
  - [ ] WaterMaterial (from Story 1.1) for visual feedback
  - [ ] VortexSystem (from Story 1.1) for activation updates
- [ ] Performance and polish
  - [ ] Ensure smooth state transitions
  - [ ] Test rapid input handling (player plays before demo ends)
  - [ ] Verify audio doesn't clip with harmony chords
  - [ ] Confirm 60 FPS during jelly animations

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Audio:** Web Audio API (native browser API)
**Scene Format:** GLB with Gaussian Splatting via Shadow Engine

**File Organization:**
```
vimana/
├── src/
│   ├── audio/
│   │   ├── HarmonyChord.ts
│   │   └── NoteFrequencies.ts (from Story 1.3)
│   ├── entities/
│   │   ├── PatientJellyManager.ts
│   │   ├── DuetProgressTracker.ts
│   │   └── JellyManager.ts (from Story 1.2)
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Teaching Sequences Detail

**Sequence 1: Lower Register Introduction**
```
Notes:    C  →  D  →  E
Strings:  0  →  1  →  2
Freqs:    261.63, 293.66, 329.63 Hz
Narrative: "The ship's voice begins low and warm. Feel the foundation."
Duration per note: ~8 seconds (demo + input)
```

**Sequence 2: Upper Register Introduction**
```
Notes:    F  →  G  →  A
Strings:  3  →  4  →  5
Freqs:    349.23, 392.00, 440.00 Hz
Narrative: "The ship's voice rises bright and hopeful. Hear the ascending light."
Duration per note: ~8 seconds (demo + input)
```

**Sequence 3: Melodic Phrase**
```
Notes:    E  →  G  →  D
Strings:  2  →  4  →  1
Freqs:    329.63, 392.00, 293.66 Hz
Narrative: "A phrase that rises and resolves. The ship speaks, and you answer."
Duration per note: ~8 seconds (demo + input)
```

### State Machine Implementation

**PatientJellyManager States:**
```javascript
enum DuetState {
    IDLE,              // Waiting to start, or between sequences
    DEMONSTRATING,     // Jelly is showing which note to play
    AWAITING_INPUT,    // Jelly submerged, waiting for player
    PLAYING_HARMONY,   // Playing success chord (brief)
    COMPLETE           // All sequences finished
}

class PatientJellyManager {
    private state: DuetState = DuetState.IDLE;
    private currentSequence: number = 0;
    private currentNoteIndex: number = 0;

    private readonly sequences = [
        [0, 1, 2], // C, D, E
        [3, 4, 5], // F, G, A
        [2, 4, 1]  // E, G, D
    ];

    startSequence(sequenceIndex: number) {
        this.currentSequence = sequenceIndex;
        this.currentNoteIndex = 0;
        this.demonstrateCurrentNote();
    }

    private demonstrateCurrentNote() {
        this.state = DuetState.DEMONSTRATING;
        const targetNote = this.sequences[this.currentSequence][this.currentNoteIndex];
        this.jellyManager.demonstrateNote(targetNote);

        // After 2 seconds, await input
        setTimeout(() => {
            this.state = DuetState.AWAITING_INPUT;
        }, 2000);
    }

    handlePlayerInput(playedNoteIndex: number) {
        if (this.state !== DuetState.AWAITING_INPUT) return;

        const targetNote = this.sequences[this.currentSequence][this.currentNoteIndex];

        if (playedNoteIndex === targetNote) {
            this.handleCorrectNote();
        } else {
            this.handleWrongNote(targetNote);
        }
    }

    private handleCorrectNote() {
        this.state = DuetState.PLAYING_HARMONY;
        this.harmonyChord.play(targetNote);

        // Advance after harmony
        setTimeout(() => {
            this.currentNoteIndex++;
            if (this.currentNoteIndex >= 3) {
                this.completeSequence();
            } else {
                this.demonstrateCurrentNote();
            }
        }, 550);
    }

    private handleWrongNote(targetNote: number) {
        // Trigger gentle feedback
        this.feedbackManager.triggerWrongNote(targetNote);

        // Stay in AWAITING_INPUT, re-demonstrate after delay
        setTimeout(() => {
            this.demonstrateCurrentNote();
        }, 1000);
    }

    private completeSequence() {
        this.completionChord.play();

        if (this.currentSequence < 2) {
            // More sequences to go
            this.currentSequence++;
            this.currentNoteIndex = 0;
            setTimeout(() => this.startSequence(this.currentSequence), 2000);
        } else {
            this.state = DuetState.COMPLETE;
            this.onDuetComplete();
        }
    }

    getProgress(): number {
        // 3 sequences × 3 notes = 9 total notes
        const completedNotes = (this.currentSequence * 3) + this.currentNoteIndex;
        return completedNotes / 9;
    }
}
```

### Harmony Chord Implementation

**HarmonyChord Class:**
```javascript
class HarmonyChord {
    private audioContext: AudioContext;
    private masterGain: GainNode;

    play(noteIndex: number) {
        const baseFreq = NOTE_FREQUENCIES[String.fromCharCode(67 + noteIndex)];
        const harmonyFreq = baseFreq * 1.5; // Perfect fifth

        // Player note oscillator
        const playerOsc = this.audioContext.createOscillator();
        const playerGain = this.audioContext.createGain();
        playerOsc.frequency.value = baseFreq;

        // Harmony note oscillator
        const harmonyOsc = this.audioContext.createOscillator();
        const harmonyGain = this.audioContext.createGain();
        harmonyOsc.frequency.value = harmonyFreq;

        // Mix: player 60%, harmony 40%
        playerGain.gain.value = 0.6;
        harmonyGain.gain.value = 0.4;

        // Envelope
        const now = this.audioContext.currentTime;
        const duration = 0.55;

        playerOsc.connect(playerGain);
        harmonyOsc.connect(harmonyGain);
        playerGain.connect(this.masterGain);
        harmonyGain.connect(this.masterGain);

        // Attack
        playerGain.gain.setValueAtTime(0, now);
        playerGain.gain.linearRampToValueAtTime(0.5, now + 0.05);
        harmonyGain.gain.setValueAtTime(0, now);
        harmonyGain.gain.linearRampToValueAtTime(0.35, now + 0.05);

        // Release
        playerGain.gain.linearRampToValueAtTime(0, now + duration);
        harmonyGain.gain.linearRampToValueAtTime(0, now + duration);

        playerOsc.start(now);
        harmonyOsc.start(now);
        playerOsc.stop(now + duration);
        harmonyOsc.stop(now + duration);
    }

    playCompletionChord() {
        // C major: C + E + G
        const freqs = [261.63, 329.63, 392.00];
        const now = this.audioContext.currentTime;
        const duration = 2.0;

        freqs.forEach((freq, i) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            osc.frequency.value = freq;
            osc.type = 'sine';

            // Stagger attacks slightly for richness
            const offset = i * 0.05;
            gain.gain.setValueAtTime(0, now + offset);
            gain.gain.linearRampToValueAtTime(0.3, now + offset + 0.1);
            gain.gain.linearRampToValueAtTime(0, now + duration);

            osc.connect(gain);
            gain.connect(this.masterGain);

            osc.start(now);
            osc.stop(now + duration);
        });
    }
}
```

### Completion Chord Frequencies

**C Major Chord (Completion):**
```
Root (C3):  261.63 Hz - 40% gain
Third (E3): 329.63 Hz - 35% gain
Fifth (G3): 392.00 Hz - 40% gain
Duration:   2.0 seconds
```

### Duet Progress Calculation

**Progress Tracker:**
```javascript
class DuetProgressTracker {
    private notesCompleted: number = 0;
    private readonly totalNotes: number = 9; // 3 sequences × 3 notes

    markNoteComplete() {
        this.notesCompleted = Math.min(this.notesCompleted + 1, this.totalNotes);
    }

    getProgress(): number {
        return this.notesCompleted / this.totalNotes;
    }

    getCurrentSequence(): number {
        return Math.floor(this.notesCompleted / 3);
    }

    getCurrentNote(): number {
        return this.notesCompleted % 3;
    }

    isComplete(): boolean {
        return this.notesCompleted >= this.totalNotes;
    }
}
```

### Memory Management

```javascript
class PatientJellyManager {
    destroy() {
        this.state = DuetState.IDLE;
        this.jellyManager.destroy();
        this.harmonyChord.destroy();
    }
}
```

### Dependencies

**Previous Story:** 1.3 Gentle Feedback (wrong note handling)

**Next Story:** 1.5 Vortex Activation (duet completion triggers vortex)

**External Dependencies:**
- JellyManager (Story 1.2)
- GentleFeedback (Story 1.3)
- WaterMaterial (Story 1.1)
- VortexSystem (Story 1.1)
- Web Audio API

### Philosophy Notes

**"This is a duet—a shared moment of music-making":**

The teaching mechanic embodies these principles:
1. **Demonstration first** - The ship shows, then asks you to join
2. **No failure** - Wrong notes are just "not yet," not "wrong"
3. **Harmony over correctness** - When you play well, the ship joins you
4. **Patient repetition** - The ship will teach forever, never gets frustrated
5. **Progressive revelation** - Each sequence builds on the last

The player should feel:
- "I'm learning a language" not "I'm being tested"
- "The ship is my teacher" not "I'm being evaluated"
- "We're making music together" not "I'm performing alone"

### References

- [Source: music-room-proto-epic.md#Story 1.4]
- [Source: gdd.md#Archive of Voices - duet philosophy]
- [Source: narrative-design.md#Musical teaching concept]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/audio/HarmonyChord.ts` (create)
- `src/entities/PatientJellyManager.ts` (create)
- `src/entities/DuetProgressTracker.ts` (create)
