# Animation Chaining & playNext - First Principles Guide

## Overview

**Animation Chaining** is the practice of sequencing multiple animations together to create complex, multi-stage movements and cinematic sequences. In the Shadow Engine, this is primarily handled through the `playNext` property, which automatically starts another animation when the current one completes.

Think of animation chaining as a **director calling "scene, take, sequence"** - just as a film is composed of many shots edited together, complex game animations are built from smaller, reusable animation clips that flow seamlessly into one another.

## What You Need to Know First

Before understanding animation chaining, you should know:
- **Basic animations** - Single animation clips from [AnimationManager](./animation-manager.md)
- **Keyframes and interpolation** - How animations calculate values
- **Animation lifecycle** - Start → Update → Complete events
- **State management** - How game state triggers animations
- **Callback patterns** - Functions that run after completion

### Quick Refresher: Why Chain Animations?

```
SINGLE ANIMATION LIMITATION:
One long animation = hard to edit, hard to reuse

┌─────────────────────────────────────────────────────────────┐
│           INTRO_CINEMATIC (12 seconds, all in one)          │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐   │
│  │shot1│shot2│shot3│shot4│shot5│shot6│shot7│shot8│shot9│   │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘   │
└─────────────────────────────────────────────────────────────┘
  Want to change shot 5? Have to edit the whole thing!

CHAINED ANIMATIONS:
Modular, reusable, easy to edit

┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ shot_1  │ → │ shot_2  │ → │ shot_3  │ → │ shot_4  │
│ (3 sec) │   │ (2 sec) │   │ (4 sec) │   │ (3 sec) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
      ↓             ↓             ↓             ↓
   playNext     playNext     playNext        (end)
  "shot_2"     "shot_3"     "shot_4"

  Want to change shot 3? Just edit that one file!
  Want to reuse shot 2? Reference it anywhere!
```

---

## Part 1: The playNext Property

### Basic Syntax

```javascript
// In animationData.js
export const animations = {
  // First animation in sequence
  intro_Part1: {
    type: "camera",
    duration: 5000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 20 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeOutCubic",
    playNext: "intro_Part2",  // ← Key: Chains to next animation
    criteria: { currentState: INTRO_START }
  },

  // Automatically plays after Part 1 finishes
  intro_Part2: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 5, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine",
    playNext: "intro_Part3"
  },

  // Final animation in sequence (no playNext = sequence ends)
  intro_Part3: {
    type: "camera",
    duration: 3000,
    keyframes: [
      { time: 0, position: { x: 5, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 2 }, target: { x: 0, y: 1.6, z: -1 } }
    ],
    easing: "easeInCubic"
    // No playNext → Chain ends, control returns to player
  }
};
```

### How playNext Works Internally

```javascript
class AnimationManager {
  play(animationKey) {
    const animation = this.animations[animationKey];
    const instance = this.createAnimationInstance(animation);

    // Set up completion handler
    instance.on('complete', () => {
      this.onAnimationComplete(animation);
    });

    this.currentAnimation = instance;
    instance.play();
  }

  onAnimationComplete(animation) {
    // Check if there's a next animation to play
    if (animation.playNext) {
      // Automatically start the next animation
      this.play(animation.playNext);
    } else {
      // End of chain - notify listeners
      this.emit('chainComplete', {
        chain: this.currentChain,
        finalAnimation: animation.key
      });
    }
  }
}
```

---

## Part 2: Chain Patterns

### 1. Linear Chain (Sequential)

The simplest pattern - animations play one after another in order:

```javascript
// Linear chain: A → B → C → D → END
export const animations = {
  sequence_A: {
    duration: 2000,
    keyframes: [/* ... */],
    playNext: "sequence_B"
  },
  sequence_B: {
    duration: 2000,
    keyframes: [/* ... */],
    playNext: "sequence_C"
  },
  sequence_C: {
    duration: 2000,
    keyframes: [/* ... */],
    playNext: "sequence_D"
  },
  sequence_D: {
    duration: 2000,
    keyframes: [/* ... */]
    // No playNext - chain ends
  }
};
```

**Use when:** You have a fixed sequence that always plays the same way.

### 2. Branching Chain (Conditional)

Different paths based on game state or player choice:

```javascript
// Branching chain: A → (B or C or D) → END
export const animations = {
  // Decision point animation
  choiceMoment: {
    type: "camera",
    duration: 3000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 2 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeOutCubic",
    // playNext set dynamically based on player choice
    playNext: null,  // Set programmatically
    criteria: { currentState: PLAYER_CHOICE_MOMENT }
  },

  // Path A: Player chose "examine"
  choice_examine: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 2 }, target: { x: -2, y: 1, z: 0 } },
      { time: 1, position: { x: -1, y: 1.5, z: 1 }, target: { x: -2, y: 1, z: 0 } }
    ],
    easing: "easeInOutCubic",
    criteria: { playerChoice: "examine" }
  },

  // Path B: Player chose "approach"
  choice_approach: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 2 }, target: { x: 0, y: 1, z: -5 } },
      { time: 1, position: { x: 0, y: 1.6, z: 0 }, target: { x: 0, y: 1.6, z: -5 } }
    ],
    easing: "easeInOutCubic",
    criteria: { playerChoice: "approach" }
  },

  // Path C: Player chose "leave"
  choice_leave: {
    type: "camera",
    duration: 3000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 2 }, target: { x: 5, y: 1, z: 5 } },
      { time: 1, position: { x: 2, y: 1.6, z: 5 }, target: { x: 10, y: 1, z: 10 } }
    ],
    easing: "easeInCubic",
    criteria: { playerChoice: "leave" }
  }
};

// Setting up dynamic branching
function onPlayerChoice(choice) {
  const choiceMoment = animations.choiceMoment;

  // Set the next animation based on choice
  switch (choice) {
    case 'examine':
      choiceMoment.playNext = 'choice_examine';
      break;
    case 'approach':
      choiceMoment.playNext = 'choice_approach';
      break;
    case 'leave':
      choiceMoment.playNext = 'choice_leave';
      break;
  }

  // Start the sequence
  animationManager.play('choiceMoment');
}
```

### 3. Looping Chain

Repeated sequence until a condition is met:

```javascript
// Looping chain: A → B → C → B → C → B → ... → (condition) → D
export const animations = {
  // Initial approach
  loopStart: {
    type: "camera",
    duration: 3000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeOutCubic",
    playNext: "loopOrbitLeft"
  },

  // Part of the loop (orbit left)
  loopOrbitLeft: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: -5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine",
    playNext: "loopOrbitRight"
  },

  // Part of the loop (orbit right)
  loopOrbitRight: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: -5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine",
    // playNext determined dynamically
    playNext: null
  },

  // Exit the loop when player interacts
  loopExit: {
    type: "camera",
    duration: 2000,
    keyframes: [
      { time: 0, position: { x: 5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 2, y: 1.6, z: 2 }, target: { x: 0, y: 1.5, z: 0 } }
    ],
    easing: "easeInCubic",
    criteria: { playerInteracted: true }
  }
};

// Dynamic loop control
class LoopController {
  constructor() {
    this.loopCount = 0;
    this.maxLoops = 3;
  }

  getNextAnimation(currentAnimation) {
    if (currentAnimation === 'loopOrbitRight') {
      this.loopCount++;

      // Check if we should exit the loop
      if (this.shouldExitLoop()) {
        return 'loopExit';
      }
    }

    // Continue the loop pattern
    if (currentAnimation === 'loopStart') {
      return 'loopOrbitLeft';
    } else if (currentAnimation === 'loopOrbitLeft') {
      return 'loopOrbitRight';
    } else if (currentAnimation === 'loopOrbitRight') {
      return 'loopOrbitLeft';
    }
  }

  shouldExitLoop() {
    // Exit after max loops OR if player interacts
    return this.loopCount >= this.maxLoops ||
           gameState.get('playerInteracted');
  }
}
```

### 4. Parallel Chain (Multiple Simultaneous)

Multiple animation chains running at the same time:

```javascript
// Parallel chains: Camera chain + Object chain + Light chain
export const animations = {
  // Chain 1: Camera movement
  camera_dollyIn: {
    type: "camera",
    duration: 5000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeOutCubic",
    playNext: "camera_orbit"
  },

  camera_orbit: {
    type: "camera",
    duration: 6000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine"
  },

  // Chain 2: Object animation (runs in parallel)
  object_reveal: {
    type: "object",
    target: "mysteriousArtifact",
    duration: 5000,
    keyframes: [
      { time: 0, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 0, y: 0, z: 0 } },
      { time: 1, rotation: { x: 0, y: Math.PI * 2, z: 0 }, scale: { x: 1, y: 1, z: 1 } }
    ],
    easing: "easeOutBack",
    playNext: "object_float"
  },

  object_float: {
    type: "object",
    target: "mysteriousArtifact",
    duration: 6000,
    loop: true,
    keyframes: [
      { time: 0, position: { x: 0, y: 1, z: 0 } },
      { time: 0.5, position: { x: 0, y: 1.5, z: 0 } },
      { time: 1, position: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine"
  },

  // Chain 3: Light animation (runs in parallel)
  light_flicker: {
    type: "light",
    target: "ambientLight",
    duration: 1000,
    loop: true,
    keyframes: [
      { time: 0, intensity: 0.5 },
      { time: 0.1, intensity: 1.0 },
      { time: 0.2, intensity: 0.3 },
      { time: 0.3, intensity: 0.8 },
      { time: 0.5, intensity: 0.5 }
    ]
  }
};

// Starting parallel chains
function startCinematic() {
  // Start all chains simultaneously
  animationManager.play('camera_dollyIn');
  animationManager.play('object_reveal');
  animationManager.play('light_flicker');

  // They run independently but are choreographed to work together
}
```

---

## Part 3: Transitions Between Animations

### Smooth Position Transitions

When chaining camera animations, ensure end positions match start positions:

```javascript
// ❌ WRONG: Discontinuous jump
part1: {
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 } },
    { time: 1, position: { x: 5, y: 1.6, z: 0 } }  // Ends here
  ]
},
part2: {
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 10 } },  // JUMP! Not continuous
    { time: 1, position: { x: -5, y: 1.6, z: 0 } }
  ]
}

// ✅ CORRECT: Continuous flow
part1: {
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 } },
    { time: 1, position: { x: 5, y: 1.6, z: 0 } }  // Ends at (5, 1.6, 0)
  ]
},
part2: {
  keyframes: [
    { time: 0, position: { x: 5, y: 1.6, z: 0 } },  // Starts where part1 ended
    { time: 1, position: { x: -5, y: 1.6, z: 0 } }
  ]
}
```

### Easing Transitions

Match easing at transition points for smooth flow:

```javascript
// Outgoing animation uses "easeOut" → decelerating at end
part1: {
  keyframes: [/* ... */],
  easing: "easeOutCubic",  // Slows down at end
  playNext: "part2"
},

// Incoming animation uses "easeIn" → accelerating from start
part2: {
  keyframes: [/* ... */],
  easing: "easeInCubic"  // Speeds up from start
}

// Result: Smooth transition with matching velocity curves
```

### Blend Duration (Overlap)

For even smoother transitions, overlap animations slightly:

```javascript
class BlendingAnimationManager {
  constructor() {
    this.blendTime = 500;  // 500ms overlap
  }

  playWithBlend(animationKey) {
    const current = this.currentAnimation;
    const next = this.animations[animationKey];

    if (current) {
      // Start new animation before current fully ends
      setTimeout(() => {
        this.play(animationKey);
      }, current.duration - this.blendTime);
    } else {
      this.play(animationKey);
    }
  }
}
```

---

## Part 4: Chain Control API

### Programmatic Chain Manipulation

```javascript
class AnimationChainController {
  constructor(animationManager) {
    this.manager = animationManager;
    this.currentChain = [];
    this.chainIndex = 0;
  }

  // Define a chain programmatically
  defineChain(name, animationKeys) {
    this.chains = this.chains || {};

    // Link animations together
    for (let i = 0; i < animationKeys.length - 1; i++) {
      const current = this.manager.animations[animationKeys[i]];
      const next = animationKeys[i + 1];
      current.playNext = next;
    }

    // Store chain for reference
    this.chains[name] = {
      animations: animationKeys,
      currentIndex: 0
    };
  }

  // Play a chain from the beginning
  playChain(name) {
    const chain = this.chains[name];
    if (!chain) {
      console.error(`Chain "${name}" not found`);
      return;
    }

    this.currentChain = name;
    this.chainIndex = 0;
    this.manager.play(chain.animations[0]);
  }

  // Jump to specific animation in chain
  jumpTo(animationKey) {
    // Stop current animation
    if (this.currentAnimation) {
      this.currentAnimation.stop();
    }

    // Start from specified point
    this.manager.play(animationKey);
  }

  // Skip to next animation in chain
  skipNext() {
    const chain = this.chains[this.currentChain];
    if (chain && this.chainIndex < chain.animations.length - 1) {
      this.currentAnimation.stop();
      this.chainIndex++;
      this.manager.play(chain.animations[this.chainIndex]);
    }
  }

  // Get current chain position
  getPosition() {
    const chain = this.chains[this.currentChain];
    return {
      chain: this.currentChain,
      animation: chain?.animations[this.chainIndex],
      index: this.chainIndex,
      total: chain?.animations.length
    };
  }
}
```

### Chain Events

```javascript
// Listen to chain lifecycle events
chainController.on('chainStart', (data) => {
  console.log(`Starting chain: ${data.chain}`);
  // Disable player input during cinematic
  gameState.set('controlEnabled', false);
});

chainController.on('animationProgress', (data) => {
  console.log(`Playing ${data.animation} (${data.index + 1}/${data.total})`);
});

chainController.on('chainComplete', (data) => {
  console.log(`Chain complete: ${data.chain}`);
  // Re-enable player input
  gameState.set('controlEnabled', true);
});

chainController.on('chainSkipped', (data) => {
  console.log(`Player skipped at ${data.skippedFrom}`);
  // Handle player skipping cinematic
});
```

---

## Part 5: Practical Chain Examples

### Example 1: Opening Cinematic

```javascript
export const openingCinematic = {
  // Fade from black, establish wide shot
  opening_wideShot: {
    type: "camera",
    duration: 5000,
    keyframes: [
      {
        time: 0,
        position: { x: 0, y: 8, z: 25 },
        target: { x: 0, y: 2, z: 0 },
        fov: 60
      },
      {
        time: 1,
        position: { x: 0, y: 5, z: 20 },
        target: { x: 0, y: 2, z: 0 },
        fov: 60
      }
    ],
    easing: "easeOutCubic",
    playNext: "opening_craneDown"
  },

  // Crane down toward scene
  opening_craneDown: {
    type: "camera",
    duration: 4000,
    keyframes: [
      {
        time: 0,
        position: { x: 0, y: 5, z: 20 },
        target: { x: 0, y: 2, z: 0 },
        fov: 60
      },
      {
        time: 1,
        position: { x: 0, y: 2, z: 15 },
        target: { x: 0, y: 1.5, z: 0 },
        fov: 60
      }
    ],
    easing: "easeInOutCubic",
    playNext: "opening_dollyIn"
  },

  // Dolly in to introduce main character/area
  opening_dollyIn: {
    type: "camera",
    duration: 3000,
    keyframes: [
      {
        time: 0,
        position: { x: 0, y: 2, z: 15 },
        target: { x: 0, y: 1.5, z: 0 },
        fov: 60
      },
      {
        time: 1,
        position: { x: 0, y: 1.7, z: 5 },
        target: { x: 0, y: 1.6, z: -1 },
        fov: 55
      }
    ],
    easing: "easeInCubic"
    // No playNext - transfer control to player
  }
};
```

### Example 2: Dialog Sequence

```javascript
export const dialogAnimations = {
  // Focus on character speaking
  dialog_focusCharacter: {
    type: "camera",
    duration: 1000,
    keyframes: [
      {
        time: 0,
        position: { x: 0, y: 1.6, z: 3 },
        target: { x: 0, y: 1.5, z: -1 }
      },
      {
        time: 1,
        position: { x: -1, y: 1.6, z: 2 },
        target: { x: -1, y: 1.5, z: -3 }
      }
    ],
    easing: "easeInOutSine",
    criteria: { currentSpeaker: "npc" },
    playNext: "dialog_listen"
  },

  // Hold while dialog plays
  dialog_listen: {
    type: "camera",
    duration: 5000,
    keyframes: [
      {
        time: 0,
        position: { x: -1, y: 1.6, z: 2 },
        target: { x: -1, y: 1.5, z: -3 }
      }
    ],
    // Very subtle breathing motion
    loop: true,
    loopDuration: 3000,
    loopKeyframes: [
      { time: 0, position: { x: -1, y: 1.6, z: 2 } },
      { time: 0.5, position: { x: -1, y: 1.62, z: 2.02 } },
      { time: 1, position: { x: -1, y: 1.6, z: 2 } }
    ],
    easing: "easeInOutSine"
    // playNext set when dialog completes
  },

  // Response from player
  dialog_playerResponse: {
    type: "camera",
    duration: 800,
    keyframes: [
      {
        time: 0,
        position: { x: -1, y: 1.6, z: 2 },
        target: { x: -1, y: 1.5, z: -3 }
      },
      {
        time: 1,
        position: { x: 0, y: 1.6, z: 0 },
        target: { x: 0, y: 1.6, z: -2 }
      }
    ],
    easing: "easeInOutSine",
    criteria: { playerResponding: true }
  }
};

// Dialog system triggers animation changes
function onDialogComplete(dialogData) {
  if (dialogData.hasNext) {
    // Continue to next dialog
    animations.dialog_listen.playNext = "dialog_nextCharacter";
  } else {
    // End dialog sequence
    animations.dialog_listen.playNext = "dialog_end";
  }
}
```

### Example 3: Building Tension

```javascript
export const tensionAnimations = {
  // Subtle hint
  tension_subtle: {
    type: "camera",
    duration: 3000,
    keyframes: [
      {
        time: 0,
        position: { x: 0, y: 1.6, z: 0 },
        target: { x: 0, y: 1.5, z: -5 }
      },
      {
        time: 1,
        position: { x: 0.5, y: 1.6, z: 0 },
        target: { x: 0, y: 1.5, z: -5 }
      }
    ],
    easing: "easeInOutSine",
    playNext: "tension_notice"
  },

  // Player notices something
  tension_notice: {
    type: "camera",
    duration: 2000,
    keyframes: [
      {
        time: 0,
        position: { x: 0.5, y: 1.6, z: 0 },
        target: { x: 0, y: 1.5, z: -5 }
      },
      {
        time: 1,
        position: { x: 0.5, y: 1.55, z: -0.5 },
        target: { x: 2, y: 2, z: -5 }
      }
    ],
    easing: "easeInOutCubic",
    playNext: "tension_focus"
  },

  // Focus on the threat
  tension_focus: {
    type: "camera",
    duration: 4000,
    keyframes: [
      {
        time: 0,
        position: { x: 0.5, y: 1.55, z: -0.5 },
        target: { x: 2, y: 2, z: -5 }
      },
      {
        time: 1,
        position: { x: 1, y: 1.6, z: 1 },
        target: { x: 5, y: 2, z: -8 }
      }
    ],
    easing: "easeInCubic",
    playNext: "tension_escalate"
  },

  // Tension builds
  tension_escalate: {
    type: "camera",
    duration: 2000,
    keyframes: [
      {
        time: 0,
        position: { x: 1, y: 1.6, z: 1 },
        target: { x: 5, y: 2, z: -8 },
        fov: 55
      },
      {
        time: 0.5,
        position: { x: 1.2, y: 1.6, z: 1 },
        target: { x: 5, y: 2, z: -8 },
        fov: 70
      },
      {
        time: 1,
        position: { x: 1, y: 1.6, z: 1 },
        target: { x: 5, y: 2, z: -8 },
        fov: 55
      }
    ],
    easing: "linear",
    loop: true
    // Continues until player acts
  }
};
```

---

## Part 6: Advanced Techniques

### Dynamic Chain Generation

Create chains based on runtime conditions:

```javascript
class DynamicChainBuilder {
  buildInspectionChain(targetObject, inspectionPoints) {
    const chain = [];

    // Starting position (current camera)
    chain.push({
      name: 'inspect_start',
      duration: 1000,
      keyframes: [{
        time: 0,
        position: camera.position.clone(),
        target: targetObject.position
      }],
      easing: "easeInOutSine"
    });

    // Add inspection points
    inspectionPoints.forEach((point, index) => {
      const prev = index === 0 ? targetObject.position : inspectionPoints[index - 1];
      chain.push({
        name: `inspect_point_${index}`,
        duration: 2000,
        keyframes: [{
          time: 0,
          position: this.calculateCameraPosition(prev, point),
          target: point
        }, {
          time: 1,
          position: this.calculateCameraPosition(point, point),
          target: point
        }],
        easing: "easeInOutCubic",
        playNext: index < inspectionPoints.length - 1
          ? `inspect_point_${index + 1}`
          : 'inspect_end'
      });
    });

    // Return to player view
    chain.push({
      name: 'inspect_end',
      duration: 1500,
      keyframes: [{
        time: 0,
        position: this.calculateCameraPosition(
          inspectionPoints[inspectionPoints.length - 1],
          targetObject.position
        ),
        target: targetObject.position
      }, {
        time: 1,
        position: { x: 0, y: 1.6, z: 0 },
        target: { x: 0, y: 1.6, z: -1 }
      }],
      easing: "easeInCubic"
    });

    return chain;
  }

  calculateCameraPosition(fromPoint, toPoint) {
    // Position camera for optimal view of target
    const direction = new THREE.Vector3()
      .subVectors(fromPoint, toPoint)
      .normalize();
    const distance = 3;
    return {
      x: toPoint.x + direction.x * distance,
      y: Math.max(1.5, toPoint.y),
      z: toPoint.z + direction.z * distance
    };
  }
}
```

### Chain State Persistence

Maintain state across chain animations:

```javascript
class ChainStateManager {
  constructor() {
    this.chainState = new Map();
  }

  setState(key, value) {
    this.chainState.set(key, value);
  }

  getState(key) {
    return this.chainState.get(key);
  }

  // Pass state between animations in a chain
  passState(fromAnimation, toAnimation, key) {
    const value = this.getState(`${fromAnimation}_${key}`);
    if (value !== undefined) {
      this.setState(`${toAnimation}_${key}`, value);
    }
  }

  // Example: Track cumulative camera rotation
  updateRotationAccumulator(animationName, rotationDelta) {
    const current = this.getState('rotation_accumulator') || 0;
    const newValue = current + rotationDelta;
    this.setState('rotation_accumulator', newValue);
    this.setState(`${animationName}_rotation`, newValue);
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Discontinuous Positions

```javascript
// ❌ WRONG: Jump between animations
part1: {
  keyframes: [{ time: 1, position: { x: 5, y: 1.6, z: 0 } }]
},
part2: {
  keyframes: [{ time: 0, position: { x: 0, y: 1.6, z: 5 } }]  // Jump!
}

// ✅ CORRECT: Match positions
part1: {
  keyframes: [{ time: 1, position: { x: 5, y: 1.6, z: 0 } }]
},
part2: {
  keyframes: [{ time: 0, position: { x: 5, y: 1.6, z: 0 } }]  // Continuous
}
```

### 2. Infinite Loops Without Exit

```javascript
// ❌ WRONG: Can't exit the loop
loopAnimation: {
  keyframes: [/* ... */],
  loop: true,
  playNext: "loopAnimation"  // Never escapes!
}

// ✅ CORRECT: Provide exit condition
loopAnimation: {
  keyframes: [/* ... */],
  loop: true,
  // playNext determined dynamically based on condition
  playNext: null  // Set based on game state
}
```

### 3. Mismatched Easing

```javascript
// ❌ WRONG: Jarring transition
part1: { easing: "easeInCubic" },  // Fast at end
part2: { easing: "easeInCubic" }   // Fast at start → double-fast transition!

// ✅ CORRECT: Match velocities
part1: { easing: "easeOutCubic" },  // Slow at end
part2: { easing: "easeInCubic" }    // Slow at start → smooth!
```

### 4. No Skip Mechanism

```javascript
// ❌ WRONG: Player forced to watch whole cinematic
function playCinematic() {
  animationManager.play('long_intro');
  // No way to skip!
}

// ✅ CORRECT: Allow skipping
function playCinematic() {
  cinemticActive = true;
  animationManager.play('long_intro');

  // Listen for skip input
  inputManager.on('skip', () => {
    if (cinematicActive) {
      animationManager.skipToEnd('long_intro');
    }
  });
}
```

---

## Performance Considerations

### Chain Optimization

```javascript
// Pre-compile chains for faster execution
class ChainCompiler {
  compileChain(animationKeys) {
    const compiled = {
      totalDuration: 0,
      animations: [],
      transitions: []
    };

    let accumulatedDuration = 0;

    animationKeys.forEach((key, index) => {
      const animation = this.animations[key];

      compiled.animations.push({
        key: key,
        offset: accumulatedDuration,
        data: animation
      });

      accumulatedDuration += animation.duration;
      compiled.totalDuration = accumulatedDuration;

      if (index < animationKeys.length - 1) {
        compiled.transitions.push({
          from: key,
          to: animationKeys[index + 1],
          at: accumulatedDuration
        });
      }
    });

    return compiled;
  }
}
```

### Memory Management

```javascript
// Clean up completed chains
function onChainComplete(chain) {
  // Remove from active set
  activeChains.delete(chain.name);

  // Dispose of any created resources
  chain.animations.forEach(anim => {
    if (anim.dispose) {
      anim.dispose();
    }
  });
}
```

---

## 🎮 Game Design Perspective

### Pacing with Chains

```javascript
// Principle: Vary pacing for emotional effect

// Fast pace - action, urgency
actionSequence: {
  part1: { duration: 1000, easing: "easeInQuad" },
  part2: { duration: 800, easing: "easeInQuad" },
  part3: { duration: 600, easing: "easeInQuad" }
  // Accelerating = building tension
}

// Slow pace - exploration, contemplation
explorationSequence: {
  part1: { duration: 5000, easing: "easeInOutSine" },
  part2: { duration: 5000, easing: "easeInOutSine" },
  part3: { duration: 5000, easing: "easeInOutSine" }
  // Consistent = meditative
}

// Build to climax - gradual acceleration
tensionBuilder: {
  part1: { duration: 4000, easing: "easeInOutSine" },
  part2: { duration: 3000, easing: "easeInOutCubic" },
  part3: { duration: 2000, easing: "easeInOutQuad" },
  part4: { duration: 1000, easing: "easeInCubic" }
  // Shortening = approaching climax
}
```

---

## Next Steps

Now that you understand animation chaining:

- [AnimationManager](./animation-manager.md) - Core animation system
- [Head-Pose Animation](./head-pose-animation.md) - Reactive character animations
- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects coordination
- [Game State System](../02-core-architecture/game-state-system.md) - State-driven animation triggers

---

## References

- [Three.js Animation System](https://threejs.org/docs/#api/en/animation/AnimationMixer) - Core animation concepts
- [Animation Best Practices](https://www.youtube.com/watch?v=Kz2M1dudq_Y) - Game animation principles
- [Cinematic Techniques](https://www.filmsite.org/filmterms.html) - Film cinematography concepts

*Documentation last updated: January 12, 2026*
