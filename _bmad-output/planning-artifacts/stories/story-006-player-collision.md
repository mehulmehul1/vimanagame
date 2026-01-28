# STORY-006: Player Collision & Water Displacement

**Epic**: `EPIC-001` - WaterBall Fluid Simulation System
**Story ID**: `STORY-006`
**Points**: `5`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **water to react to my presence and movement**, so that **I feel physically connected to the fluid world I'm exploring**.

---

## Overview

Implement player-water interaction where the player's body displaces fluid particles and creates wake effects. Similar to how WaterBall handles mouse interaction, but with a 3D cylindrical collider representing the player.

**Reference:**
- [WaterBall mouse interaction](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125) (mouseInfoUniformBuffer)
- [updateGrid.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/updateGrid.wgsl) (mouse interaction shader)

---

## Technical Specification

### Player Collision Model

```
┌──────────────────────────────────────────────────────────────┐
│  Player Movement → Velocity Field → Particle Displacement     │
│                                                               │
│  1. Get player position/velocity from CharacterController    │
│  2. Upload to GPU as uniform (similar to mouse interaction)   │
│  3. Apply displacement force to particles within radius      │
│  4. Particles push aside, create wake behind player          │
└──────────────────────────────────────────────────────────────┘
```

### Player Interaction Data Structure

```typescript
interface PlayerWaterInteraction {
    // Player state
    position: vec3;           // Current world position
    velocity: vec3;           // Current movement velocity
    previousPosition: vec3;   // For wake direction calculation

    // Collision geometry
    colliderRadius: number;   // Default: 0.8 (player capsule radius)
    colliderHeight: number;   // Default: 1.8 (player height)
    buoyancyFactor: number;   // How much water pushes player (default: 0.3)

    // Water interaction
    isInWater: boolean;       // Player feet below water surface
    waterSurfaceY: number;    // Y-level of water surface
    immersionDepth: number;   // How deep player is in water (0-1)
}
```

---

## Implementation Tasks

1. **[SYSTEM]** Create `PlayerWaterInteraction` system class
2. **[UNIFORM]** Add `playerInteraction` uniform buffer (position, velocity, radius)
3. **[COMPUTE]** Modify `updateGrid.wgsl` to accept player interaction
4. **[DETECTION]** Raycast/height check to detect if player is in water
5. **[PHYSICS]** Apply buoyancy force to CharacterController when in water
6. **[WAKE]** Create wake trail effect based on player movement direction
7. **[AUDIO]** Trigger water splash sounds on entry/exit

---

## File Structure

```
src/systems/fluid/interaction/
├── PlayerWaterInteraction.ts       # Main player-water system
├── PlayerWakeEffect.ts             # Wake VFX
└── shaders/
    └── updateGridWithPlayer.wgsl   # Modified updateGrid shader
```

---

## PlayerWaterInteraction Interface

```typescript
export class PlayerWaterInteraction {
    private interaction: PlayerWaterInteraction;
    private interactionBuffer: GPUBuffer;

    constructor(
        private device: GPUDevice,
        private particleSystem: MLSMPMSimulator,
        private characterController: CharacterController
    ) {
        // Create uniform buffer for player interaction
        this.interactionBuffer = device.createBuffer({
            size: 4 * 4, // position (3) + radius (1)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.interaction = {
            position: [0, 0, 0],
            velocity: [0, 0, 0],
            previousPosition: [0, 0, 0],
            colliderRadius: 0.8,
            colliderHeight: 1.8,
            buoyancyFactor: 0.3,
            isInWater: false,
            waterSurfaceY: 0.0, // ArenaFloor water level
            immersionDepth: 0.0,
        };
    }

    // Call each frame to update interaction
    update(deltaTime: number): void {
        // Get player state from CharacterController
        const playerPos = this.characterController.getPosition();
        const playerVel = this.characterController.getVelocity();

        // Update interaction state
        this.interaction.previousPosition = [...this.interaction.position];
        this.interaction.position = playerPos;
        this.interaction.velocity = playerVel;

        // Check if player is in water
        const feetY = playerPos[1] - 0.9; // Approximate feet position
        this.interaction.isInWater = feetY < this.interaction.waterSurfaceY;

        // Calculate immersion depth (0 = dry, 1 = fully submerged)
        if (this.interaction.isInWater) {
            const submergedAmount = this.interaction.waterSurfaceY - feetY;
            this.interaction.immersionDepth = Math.min(submergedAmount / 1.8, 1.0);
        } else {
            this.interaction.immersionDepth = 0.0;
        }

        // Apply buoyancy to player if in water
        if (this.interaction.isInWater) {
            this.applyBuoyancy();
        }

        // Upload interaction data to GPU
        this.uploadInteractionData();
    }

    private applyBuoyancy(): void {
        // Apply upward force based on immersion depth
        const buoyancyForce = this.interaction.immersionDepth *
                              this.interaction.buoyancyFactor * 9.8;

        // Also dampen player velocity (water resistance)
        const drag = 1.0 - (this.interaction.immersionDepth * 0.1);

        this.characterController.applyForce([0, buoyancyForce, 0]);
        this.characterController.applyDrag(drag);
    }

    private uploadInteractionData(): void {
        const data = new Float32Array([
            ...this.interaction.position,  // 0-2
            this.interaction.colliderRadius, // 3
        ]);
        this.device.queue.writeBuffer(this.interactionBuffer, 0, data);
    }

    // For wake effect system
    getPlayerMovementDirection(): vec3 {
        const delta = [
            this.interaction.position[0] - this.interaction.previousPosition[0],
            this.interaction.position[1] - this.interaction.previousPosition[1],
            this.interaction.position[2] - this.interaction.previousPosition[2],
        ] as vec3;
        return normalize(delta);
    }

    getPlayerSpeed(): number {
        return length(this.interaction.velocity);
    }
}
```

---

## Shader Integration (updateGrid.wgsl)

Add to existing updateGrid shader:

```wgsl
// Player interaction uniform
struct PlayerInteraction {
    position: vec3f,
    radius: f32,
};

@group(0) @binding(7) var<uniform> player: PlayerInteraction;

// In updateGrid compute function
fn applyPlayerForces(cellPos: vec3f) {
    let toPlayer = cellPos - player.position;
    let dist = length(toPlayer);

    // Only affect cells within player radius
    if (dist < player.radius) {
        // Calculate displacement force
        // Push particles AWAY from player center
        let dirToCell = normalize(toPlayer);
        let pushStrength = (player.radius - dist) / player.radius;

        // Player movement creates additional wake
        let playerVel = vec3f(0.0, 0.0, 0.0); // Could add velocity uniform
        let wakeForce = playerVel * pushStrength * 0.5;

        // Apply forces
        let displacementForce = dirToCell * pushStrength * 2.0;
        cell.vx += i32((displacementForce.x + wakeForce.x) * fixed_point_multiplier);
        cell.vy += i32((displacementForce.y + wakeForce.y) * fixed_point_multiplier);
        cell.vz += i32((displacementForce.z + wakeForce.z) * fixed_point_multiplier);
    }
}
```

---

## Wake Effect System

Create visual wake trail behind player:

```typescript
class PlayerWakeEffect {
    private wakeParticles: WakeParticle[] = [];

    update(deltaTime: number, playerInteraction: PlayerWaterInteraction): void {
        // Only create wake if player is moving through water
        if (!playerInteraction.isInWater) return;

        const speed = playerInteraction.getPlayerSpeed();
        if (speed < 0.1) return; // Not moving enough

        // Spawn wake particles at player position
        const wakeDir = playerInteraction.getPlayerMovementDirection();
        const behindPlayer = [
            playerInteraction.position[0] - wakeDir[0] * 0.5,
            playerInteraction.waterSurfaceY, // At water surface
            playerInteraction.position[2] - wakeDir[2] * 0.5,
        ];

        this.wakeParticles.push({
            position: behindPlayer,
            radius: 0.1 + speed * 0.2,
            maxRadius: 1.0 + speed * 0.5,
            alpha: 0.6,
            age: 0,
            lifetime: 2.0,
        });

        // Update and cull old particles
        this.wakeParticles = this.wakeParticles.filter(wake => {
            wake.age += deltaTime;
            wake.radius += deltaTime * 0.5;
            wake.alpha = 0.6 * (1.0 - wake.age / wake.lifetime);
            return wake.age < wake.lifetime;
        });
    }

    render(encoder: GPUCommandEncoder): void {
        // Render wake particles as expanding rings on water surface
        // Use instanced rendering for performance
    }
}
```

---

## Audio Integration

Water sound effects based on player interaction:

```typescript
// In PlayerWaterInteraction
update(deltaTime: number): void {
    const wasInWater = this.interaction.isInWater;
    // ... (existing update code)

    // Detect water entry/exit for sounds
    if (!wasInWater && this.interaction.isInWater) {
        // Just entered water
        this.audioManager.playOneShot('water-entry', {
            volume: this.interaction.immersionDepth,
            position: this.interaction.position,
        });
    } else if (wasInWater && !this.interaction.isInWater) {
        // Just exited water
        this.audioManager.playOneShot('water-exit', {
            volume: this.getExitSpeed(),
            position: this.interaction.position,
        });
    }

    // Continuous water movement sound
    if (this.interaction.isInWater) {
        const speed = this.getPlayerSpeed();
        if (speed > 0.5) {
            this.audioManager.playLooped('water-movement', {
                volume: Math.min(speed / 5.0, 1.0),
                position: this.interaction.position,
            });
        }
    }
}
```

---

## Buoyancy Physics

When player is in water:

```typescript
// Buoyancy calculation
private applyBuoyancy(): void {
    const densityWater = 1.0;
    const densityPlayer = 0.98; // Slightly less than water (human body)
    const gravity = 9.81;

    // Volume of displaced water
    const displacedVolume = this.interaction.immersionDepth *
                           Math.PI * Math.pow(this.interaction.colliderRadius, 2) *
                           this.interaction.colliderHeight;

    // Buoyancy force = weight of displaced fluid
    const buoyancyForce = displacedVolume * densityWater * gravity;

    // Net force (buoyancy - gravity)
    const netForce = buoyancyForce - (this.getPlayerMass() * gravity);

    // Apply upward force (capped to prevent shooting out of water)
    const clampedForce = Math.max(Math.min(netForce, 500), -100);

    this.characterController.applyForce([0, clampedForce, 0]);

    // Water drag (slows movement)
    const dragCoeff = 0.5 * this.interaction.immersionDepth;
    const dragForce = -this.interaction.velocity.map(v => v * dragCoeff);
    this.characterController.applyForce(dragForce);
}
```

---

## Integration with HarpRoom

```typescript
// In HarpRoom.ts
export class HarpRoom {
    private playerWaterInteraction: PlayerWaterInteraction;
    private wakeEffect: PlayerWakeEffect;

    update(deltaTime: number): void {
        // Update player-water interaction
        this.playerWaterInteraction.update(deltaTime);

        // Update wake VFX
        this.wakeEffect.update(deltaTime, this.playerWaterInteraction);

        // Render wake
        this.wakeEffect.render(this.commandEncoder);
    }

    // For debug/UI
    getPlayerWaterState(): PlayerWaterInteraction {
        return this.playerWaterInteraction.getState();
    }
}
```

---

## Acceptance Criteria

- [ ] `PlayerWaterInteraction` class created and integrated
- [ ] Player walking through water displaces particles visibly
- [ ] Player experiences buoyancy when in water (floats upward)
- [ ] Player movement creates wake trail behind them
- [ ] Wake particles fade and expand naturally
- [ ] Water entry/exit plays appropriate audio
- [ ] Performance: Player interaction adds <0.5ms per frame
- [ ] Debug view: `window.debugVimana.fluid.getPlayerInteraction()` returns state

---

## Dependencies

- **Requires**: STORY-001 (MLS-MPM simulator)
- **Requires**: Existing CharacterController with position/velocity access
- **Requires**: AudioManager for water sounds
- **Blocks**: None (can develop in parallel with other interaction stories)

---

## Configuration Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `colliderRadius` | 0.8 | 0.5 - 1.2 | Size of player water collider |
| `buoyancyFactor` | 0.3 | 0.1 - 1.0 | How strongly water pushes player up |
| `dragCoeff` | 0.5 | 0.1 - 1.0 | Water resistance to movement |
| `wakeMaxRadius` | 1.5 | 0.5 - 3.0 | Size of wake ripples |
| `wakeLifetime` | 2.0 | 0.5 - 5.0 | How long ripples persist |

---

## Edge Cases

1. **Player jumping into water**: Detect rapid Y velocity change, play splash sound
2. **Player swimming underwater**: Apply stronger drag, limit vertical movement
3. **Player in tunnel mode**: Water forms sphere around player, adjust interaction radius
4. **Fast player movement**: Clamp displacement force to prevent particle explosion
5. **Player teleport**: Reset previousPosition to avoid wake spawn at teleport destination

---

## Notes

- **Performance**: Player interaction is similar to mouse interaction in WaterBall - very cheap (just one uniform)
- **Visual quality**: Wake effect makes player feel connected to water
- **Gameplay**: Buoyancy can be used for platforming (floating upward to reach areas)
- **Tunnel mode**: When sphere forms, player can walk through hollow center - no water displacement needed there

---

**Sources:**
- [WaterBall mouse interaction](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125)
- [WaterBall updateGrid shader](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/updateGrid.wgsl)
