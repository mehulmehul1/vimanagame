# STORY-004: Sphere Constraint Animation (Plane→Tunnel)

**Epic**: `EPIC-001` - WaterBall Fluid Simulation System
**Story ID**: `STORY-004`
**Points**: `3`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **water to transform into a tunnel portal as music progresses**, so that **I know the vortex is opening and I can enter**.

---

## Overview

Animate the simulation bounds to create a sphere constraint effect. As `uDuetProgress` goes from 0→1, the water transforms from a flat plane into a hollow sphere tunnel that the player can enter.

**Source:**
- [WaterBall g2p.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/g2p.wgsl) (sphere constraint logic)
- [WaterBall main.ts](https://github.com/matsuoka-601/WaterBall/blob/master/main.ts) (box animation)

---

## Technical Specification

### Animation States

```
uDuetProgress = 0.0:  Flat plane water (ArenaFloor)
uDuetProgress = 0.5:  Beginning to curve upward
uDuetProgress = 1.0:  Complete sphere tunnel (hollow center)
```

### Box Size Animation (from WaterBall)

```typescript
// In MLSMPMSimulator.update()
const initBoxSize = [52, 52, 52];  // Initial box dimensions
const boxWidthRatio = 0.5 + (uDuetProgress * 0.5);

// Shrink Z dimension to create sphere/tunnel
realBoxSize[2] = initBoxSize[2] * boxWidthRatio;

// The sphere constraint forces particles toward sphere surface
// As box shrinks, particles are pushed into sphere shape
```

### Animation Parameters

| Parameter | Progress 0.0 | Progress 1.0 |
|-----------|---------------|---------------|
| `boxSize.x` | 52 | 52 |
| `boxSize.y` | 52 | 52 |
| `boxSize.z` | 52 | 26 (50%) |
| `sphereRadius` | 15 | 15 |
| `tunnelOpen` | false | true |

### Sphere Constraint Math (from g2p.wgsl)

```wgsl
let center = vec3f(real_box_size.x / 2, real_box_size.y / 2, real_box_size.z / 2);
let dist = center - particles[id.x].position;
let dirToOrigin = normalize(dist);
let r: f32 = sphereRadius;

// Push particles to sphere surface (creates hollow tunnel center)
if (dot(dist, dist) < r * r) {
    particles[id.x].v += -(r - sqrt(dot(dist, dist))) * dirToOrigin * 3.0;
}

// Gentle attraction keeps particles constrained
particles[id.x].v += dirToOrigin * 0.1;
```

---

## Implementation Tasks

1. **[UNIFORM]** Add `uBoxSize` uniform to pass animated box dimensions
2. **[UNIFORM]** Add `uSphereRadius` uniform for tunnel size
3. **[COMPUTE]** Update `g2p.wgsl` with sphere constraint from WaterBall
4. **[ANIMATION]** Create `SphereConstraintAnimator` class
5. **[INTEGRATION]** Hook `uDuetProgress` changes to box size updates
6. **[TIMING]** Set animation speed: minClosingSpeed = -0.01, maxOpeningSpeed = 0.04

---

## File Structure

```
src/systems/fluid/
├── SphereConstraintAnimator.ts    # Handles plane→sphere animation
└── types.ts
```

---

## SphereConstraintAnimator Interface

```typescript
export class SphereConstraintAnimator {
    // Current animation state
    private boxWidthRatio: number = 0.5;
    private targetBoxWidthRatio: number = 0.5;

    // Animation parameters
    private readonly minClosingSpeed = -0.01;
    private readonly maxOpeningSpeed = 0.04;

    constructor(
        private simulator: MLSMPMSimulator,
        private initBoxSize: vec3
    ) {}

    // Call each frame to update animation
    update(duetProgress: number, deltaTime: number): void {
        // Map duetProgress to target box ratio
        this.targetBoxWidthRatio = 0.5 + (duetProgress * 0.5);

        // Animate toward target (smooth interpolation)
        const dVal = this.targetBoxWidthRatio - this.boxWidthRatio;
        const clamped = Math.max(dVal, this.minClosingSpeed);
        const adjusted = Math.min(clamped, this.maxOpeningSpeed);

        this.boxWidthRatio += adjusted;

        // Update simulator box size
        const realBoxSize = [
            this.initBoxSize[0],
            this.initBoxSize[1],
            this.initBoxSize[2] * this.boxWidthRatio
        ];

        this.simulator.changeBoxSize(realBoxSize);
    }

    // Get current tunnel radius for gameplay
    getTunnelRadius(): number {
        return this.simulator.sphereRadius;
    }

    // Is tunnel fully open (player can enter)?
    isTunnelOpen(): boolean {
        return this.boxWidthRatio >= 0.95;
    }
}
```

---

## Integration with HarpRoom

```typescript
// In HarpRoom.ts
export class HarpRoom {
    private sphereAnimator: SphereConstraintAnimator;

    updateDuetProgress(progress: number): void {
        this.duetProgress = progress;

        // Animate water toward tunnel shape
        this.sphereAnimator.update(progress, this.deltaTime);

        // Visual feedback on vortex
        if (this.sphereAnimator.isTunnelOpen()) {
            this.vortexGate.open();
        }
    }
}
```

---

## Visual Progression

```
Progress 0.0 - 0.2:
┌─────────────────────────────┐
│  Flat water surface          │
│  ~~~~~~~~~~~~~~~~~~~~~~~~      │
└─────────────────────────────┘

Progress 0.3 - 0.6:
┌─────────────────────────────┐
│        _--_                  │
│     _/      \_               │
│    |          |              │
│     \________/               │
└─────────────────────────────┘
Water beginning to curve upward

Progress 0.7 - 1.0:
    ╭───────────────╮
   ╱                 ╲
  │    (HOLLOW)       │  ← Tunnel!
   ╲                 ╱
    ╰───────────────╯
Player can enter through center
```

---

## Acceptance Criteria

- [ ] `SphereConstraintAnimator` class created and integrated
- [ ] `uDuetProgress` change animates box size smoothly
- [ ] Particles visually form sphere/tunnel shape as progress increases
- [ ] Center of sphere remains hollow (player can walk through)
- [ ] Animation speed feels natural (~5-10 seconds from start to finish)
- [ ] No particles escape the simulation bounds during animation
- [ ] Debug view: `window.debugVimana.fluid.getBoxRatio()` returns current state

---

## Dependencies

- **Requires**: STORY-001 (MLS-MPM simulator with g2p shader)
- **Requires**: Existing `uDuetProgress` in HarpRoom
- **Blocks**: STORY-006 (tunnel state needed for player interaction)

---

## Edge Cases

1. **Player inside tunnel during animation**: Clamp box size to prevent crushing player
2. **Rapid progress changes**: Use max closing/opening speed limits
3. **Performance**: Box size change is cheap (just uniform update)
4. **Particles escaping**: Wall constraint in g2p.wgsl prevents escape

---

**Sources:**
- [WaterBall main.ts line ~220](https://github.com/matsuoka-601/WaterBall/blob/master/main.ts#L220) (box animation)
- [WaterBall g2p.wgsl line ~60](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/g2p.wgsl#L60) (sphere constraint)
