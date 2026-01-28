# STORY-005: Harp-to-Water Interaction System

**Epic**: `EPIC-001` - WaterBall Fluid Simulation System
**Story ID**: `STORY-005`
**Points**: `5`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **harp strings to create waves and ripples in the water**, so that **my music feels like it's shaping the world around me**.

---

## Overview

Create a coupling system where each of the 6 harp strings transfers energy to the fluid simulation. String vibrations create velocity fields that displace particles and generate waves.

**Reference:**
- [WaterBall mouse interaction](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125) (mouseInfoUniformBuffer)
- [updateGrid.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/updateGrid.wgsl) (mouse interaction shader)

---

## Technical Specification

### Harp String Configuration

```typescript
// String positions (from HarpRoom scene scan)
const STRING_POSITIONS = [
    { id: 'String_1', index: 0, worldPos: [49.8, 2.3, -7.7] },
    { id: 'String_2', index: 1, worldPos: [49.9, 2.3, -7.3] },
    { id: 'String_3', index: 2, worldPos: [49.9, 2.4, -7.0] },
    { id: 'String_4', index: 3, worldPos: [49.9, 2.4, -6.7] },
    { id: 'String_5', index: 4, worldPos: [49.8, 2.3, -6.4] },
    { id: 'String_6', index: 5, worldPos: [49.8, 2.3, -6.1] },
];

// Musical notes
const STRING_FREQUENCIES = [
    261.63, // C4
    293.66, // D4
    329.63, // E4
    349.23, // F4
    392.00, // G4
    440.00, // A4
];
```

### Interaction Model

```
┌──────────────────────────────────────────────────────────────┐
│  String Vibration → Velocity Field → Particle Displacement     │
│                                                               │
│  1. String plucked                                           │
│  2. Get vibration amplitude (from audio/visual)                │
│  3. Map amplitude to force vector toward water surface         │
│  4. Apply force to particles within radius                     │
│  5. Particles ripple outward naturally                         │
└──────────────────────────────────────────────────────────────┘
```

### Force Calculation

```typescript
interface StringWaterInteraction {
    stringIndex: number;        // 0-5
    stringPosition: vec3;       // World-space string position
    waterSurfaceY: number;       // Y-level of water surface

    // Current vibration state
    amplitude: number;           // 0-1, vibrational intensity
    phase: number;              // Oscillation phase
    frequency: number;          // String frequency

    // Interaction parameters
    influenceRadius: number;    // How far string affects water (default: 8.0)
    forceMultiplier: number;    // Strength of interaction (default: 5.0)
}
```

---

## Implementation Tasks

1. **[SYSTEM]** Create `HarpWaterInteraction` system class
2. **[UNIFORM]** Add `stringForces` uniform buffer (6 vec3 forces)
3. **[COMPUTE]** Modify `updateGrid.wgsl` to accept string force inputs
4. **[DETECTION]** Raycast from string positions to water surface
5. **[MAPPING]** Map string vibration to force vectors
6. **[VISUAL]** Create ring/ripple VFX at string impact points
7. **[AUDIO]** Hook into existing harp audio trigger system

---

## File Structure

```
src/systems/fluid/interaction/
├── HarpWaterInteraction.ts       # Main interaction system
├── StringForceCalculator.ts      # Converts vibration to forces
└── shaders/
    └── updateGridWithStrings.wgsl # Modified updateGrid shader
```

---

## HarpWaterInteraction Interface

```typescript
export class HarpWaterInteraction {
    private interactions: Map<number, StringWaterInteraction> = new Map();
    private forceBuffer: GPUBuffer;  // 6 × vec3 = 24 floats

    constructor(
        private device: GPUDevice,
        private particleSystem: MLSMPMSimulator,
        stringPositions: vec3[]
    ) {
        // Create uniform buffer for forces
        this.forceBuffer = device.createBuffer({
            size: 6 * 3 * 4, // 6 strings × 3 components × 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    // Called when string is plucked
    onStringPlucked(stringIndex: number, intensity: number): void {
        const interaction = this.interactions.get(stringIndex);
        if (interaction) {
            interaction.amplitude = intensity;
            interaction.phase = 0;
        }
    }

    // Called each frame to update forces
    update(deltaTime: number): void {
        const forces: vec3[] = [];

        for (let i = 0; i < 6; i++) {
            const interaction = this.interactions.get(i);
            if (!interaction || interaction.amplitude <= 0.01) {
                forces.push([0, 0, 0]);
                continue;
            }

            // Calculate force based on string vibration
            const force = this.calculateStringForce(interaction);
            forces.push(force);

            // Decay amplitude
            interaction.amplitude *= 0.95;
        }

        // Upload forces to GPU
        this.device.queue.writeBuffer(this.forceBuffer, 0, new Float32Array(forces.flat()));
    }

    private calculateStringForce(interaction: StringWaterInteraction): vec3 {
        // Direction from string to water surface point
        const stringPos = interaction.stringPosition;
        const waterY = interaction.waterSurfaceY;

        // Force is primarily downward (into water)
        const forceY = -interaction.amplitude * interaction.forceMultiplier;

        // Slight spread based on string vibration pattern
        const spreadX = Math.sin(interaction.phase * 2) * interaction.amplitude * 0.3;
        const spreadZ = Math.cos(interaction.phase * 2) * interaction.amplitude * 0.3;

        return [spreadX, forceY, spreadZ];
    }
}
```

---

## Shader Integration (updateGrid.wgsl)

Add to existing updateGrid shader:

```wgsl
// New uniform for string forces
@group(0) @binding(6) var<uniform> stringForces: array<vec3f, 6>;

// In updateGrid compute function
fn applyStringForces(cellPos: vec3f) {
    // String positions in world space
    const STRING_POSITIONS = array<vec3f, 6>(
        vec3f(49.8, 2.3, -7.7),  // String 1
        vec3f(49.9, 2.3, -7.3),  // String 2
        // ... etc
    );

    const INFLUENCE_RADIUS = 8.0;

    for (var s = 0; s < 6; s++) {
        let dist = length(cellPos - STRING_POSITIONS[s]);

        if (dist < INFLUENCE_RADIUS) {
            // Apply force with falloff
            let falloff = 1.0 - (dist / INFLUENCE_RADIUS);
            let force = stringForces[s] * falloff;

            // Add to cell velocity
            cell.vx += i32(force.x * fixed_point_multiplier);
            cell.vy += i32(force.y * fixed_point_multiplier);
            cell.vz += i32(force.z * fixed_point_multiplier);
        }
    }
}
```

---

## Visual Feedback

Add ripple rings at string impact points:

```typescript
class StringRippleEffect {
    private ripples: Ripple[] = [];

    // Spawn visual ripple when string plucked
    onStringPlucked(stringIndex: number, intensity: number): void {
        this.ripples.push({
            position: STRING_POSITIONS[stringIndex],
            radius: 0.1,
            maxRadius: 3.0 + intensity * 2.0,
            alpha: 1.0,
            age: 0,
        });
    }

    // Render ripples as expanding rings
    render(encoder: GPUCommandEncoder): void {
        // Draw ring quads with instancing
        // Fade out over time
    }
}
```

---

## Integration with HarpRoom

```typescript
// In HarpRoom.ts
export class HarpRoom {
    private harpWaterInteraction: HarpWaterInteraction;

    onStringPlucked(stringIndex: number): void {
        // Existing: Play note
        this.harmonyChord.playNote(this.stringNotes[stringIndex]);

        // NEW: Create water interaction
        const intensity = this.getStringVelocity(stringIndex);
        this.harpWaterInteraction.onStringPlucked(stringIndex, intensity);
    }

    update(deltaTime: number): void {
        // NEW: Update string-water forces
        this.harpWaterInteraction.update(deltaTime);
    }
}
```

---

## Acceptance Criteria

- [ ] `HarpWaterInteraction` class created and integrated
- [ ] Each of 6 strings triggers unique ripple pattern in water
- [ ] Ripple amplitude proportional to string pluck intensity
- [ ] Ripples propagate naturally through particle system
- [ ] Visual ripple rings appear at string impact points
- [ ] Performance: String interaction adds <1ms per frame
- [ ] Debug view: `window.debugVimana.fluid.getStringForce(index)`

---

## Dependencies

- **Requires**: STORY-001 (MLS-MPM simulator)
- **Requires**: Existing harp string position data
- **Requires**: Existing harp audio trigger system
- **Blocks**: None (can develop in parallel)

---

## Configuration Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `influenceRadius` | 8.0 | 3.0 - 15.0 | How far string affects water |
| `forceMultiplier` | 5.0 | 1.0 - 20.0 | Strength of force |
| `decayRate` | 0.95 | 0.9 - 0.99 | How fast vibrations fade |
| `rippleMaxRadius` | 5.0 | 2.0 - 10.0 | Visual ripple size |

---

## Notes

- **String positions**: Hardcoded from HarpRoom scene (see console logs for exact positions)
- **Frequency mapping**: Each string has different natural frequency - can affect ripple pattern
- **Chord effects**: When multiple strings played, forces add up (superposition)
- **Tunnel mode**: When sphere forms, strings still affect water but differently (may need tuning)

---

**Sources:**
- [WaterBall mouse interaction](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125)
- [String positions from HarpRoom logs](file://./console-info.txt)
