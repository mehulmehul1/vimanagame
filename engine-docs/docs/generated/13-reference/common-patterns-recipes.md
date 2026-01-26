# Common Patterns & Recipes - Shadow Engine

## Overview

This **Common Patterns & Recipes** guide provides solutions to frequently encountered gameplay scenarios. Rather than starting from scratch, you can adapt these proven patterns for your own game.

Think of this as your **"cookbook"**—like a cookbook contains recipes for common dishes, this guide contains recipes for common gameplay mechanics: doors that open, items you can pick up, enemies that follow you, and more.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Empower developers to focus on what makes their game unique, not reinventing common mechanics. When basic patterns are easy to implement, developers spend more energy on creative innovation.

**Why Patterns Matter?**
- **Player Expectations**: Players understand familiar mechanics
- **Proven Solutions**: These patterns work well in practice
- **Quick Iteration**: Implement core gameplay faster
- **Consistency**: Maintainable and understandable code

---

## 🛠️ Pattern Format

Each pattern includes:
- **What It Does**: Brief description of the mechanic
- **When to Use It**: Appropriate use cases
- **How It Works**: Technical explanation
- **Code Example**: Ready-to-use implementation
- **Variations**: Ways to customize it

---

## INTERACTION PATTERNS

---

## Pattern: Simple Door

**What It Does**: A door that opens when the player interacts with it.

**When to Use It**: Any doorway between rooms or areas.

**How It Works**: The door has a trigger zone. When the player presses the interact key while in the zone, the door animates open.

### Code Example

```javascript
// 1. Create door data (in InteractiveObjectData.js)
export const DOOR_HOUSE = {
  id: 'door_house',
  name: 'House Door',
  type: 'door',
  model: '/assets/models/door.glb',
  position: { x: 5, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },

  trigger: {
    shape: 'box',
    size: { x: 2, y: 3, z: 1 },
    offset: { x: 0, y: 1.5, z: 0 }
  },

  interaction: {
    prompt: 'Press E to open',
    action: 'openDoor',
    cooldown: 500
  },

  // Door-specific properties
  door: {
    isOpen: false,
    locked: false,
    openAngle: Math.PI / 2,  // 90 degrees
    openDuration: 1.0,       // seconds
    autoClose: true,
    autoCloseDelay: 5        // seconds
  }
};

// 2. Handle door interaction
class DoorController {
  constructor(objectData, sceneManager, animationManager) {
    this.door = objectData;
    this.scene = sceneManager;
    this.animation = animationManager;

    // Create the door mesh
    this.mesh = this.createDoorMesh();

    // Store original rotation
    this.originalRotation = this.mesh.rotation.y;

    // Listen for interaction
    game.on('interact:' + objectData.id, () => this.onInteract());
  }

  createDoorMesh() {
    // Load or create door mesh
    const mesh = this.scene.loadModel(this.door.model);
    mesh.position.set(
      this.door.position.x,
      this.door.position.y,
      this.door.position.z
    );
    mesh.rotation.y = this.door.rotation.y;
    return mesh;
  }

  async onInteract() {
    if (this.door.door.locked) {
      // Play locked sound
      game.getManager('audio').playSFX('door_locked');
      return;
    }

    if (this.door.door.isOpen) {
      await this.close();
    } else {
      await this.open();
    }
  }

  async open() {
    this.door.door.isOpen = true;

    // Animate door opening
    await this.animation.play('door_open', {
      target: this.mesh,
      rotation: {
        y: this.originalRotation + this.door.door.openAngle
      },
      duration: this.door.door.openDuration,
      easing: 'easeOutCubic'
    });

    // Play open sound
    game.getManager('audio').playSFX('door_open');

    // Schedule auto-close
    if (this.door.door.autoClose) {
      setTimeout(() => {
        if (this.door.door.isOpen) {
          this.close();
        }
      }, this.door.door.autoCloseDelay * 1000);
    }
  }

  async close() {
    this.door.door.isOpen = false;

    await this.animation.play('door_close', {
      target: this.mesh,
      rotation: {
        y: this.originalRotation
      },
      duration: this.door.door.openDuration,
      easing: 'easeInOutCubic'
    });

    game.getManager('audio').playSFX('door_close');
  }
}
```

### Variations

- **Sliding Door**: Change animation to move position instead of rotation
- **Locked Door**: Add key requirement to open
- **One-Way Door**: Only opens from one side
- **Double Door**: Two mesh halves that swing apart

---

## Pattern: Pick-Up Item

**What It Does**: An item the player can collect by walking over it.

**When to Use It**: Collectibles, health packs, ammo, keys.

**How It Works**: The item has a trigger zone. When the player enters, the item animates toward the player, disappears, and is added to inventory.

### Code Example

```javascript
// 1. Create item data
export const ITEM_HEALTH_PACK = {
  id: 'health_pack_01',
  name: 'Health Pack',
  type: 'pickup',
  model: '/assets/models/health_pack.glb',
  position: { x: 3, y: 0.5, z: 2 },

  trigger: {
    shape: 'sphere',
    radius: 1.5
  },

  pickup: {
    itemType: 'health',
    amount: 25,
    respawn: false,
    respawnTime: 60  // seconds if enabled
  },

  effects: {
    spawn: {
      type: 'float',
      amplitude: 0.2,
      frequency: 2
    },
    collect: {
      type: 'implode',
      duration: 0.3
    }
  }
};

// 2. Item pickup controller
class PickupController {
  constructor(itemData, sceneManager, audio, vfx) {
    this.item = itemData;
    this.scene = sceneManager;
    this.audio = audio;
    this.vfx = vfx;

    this.mesh = null;
    this.active = true;
    this.spawnTime = Date.now();
  }

  async spawn() {
    // Create the item mesh
    this.mesh = await this.scene.loadModel(this.item.model);
    this.mesh.position.set(
      this.item.position.x,
      this.item.position.y,
      this.item.position.z
    );

    // Add floating animation
    this.startFloatingEffect();

    // Register trigger
    this.scene.registerTrigger(this.item.id, {
      shape: this.item.trigger.shape,
      ...this.item.trigger,
      onEnter: (entity) => this.onPlayerEnter(entity)
    });
  }

  startFloatingEffect() {
    const baseY = this.item.position.y;
    const amplitude = this.item.effects.spawn.amplitude;
    const frequency = this.item.effects.spawn.frequency;

    // Animate floating
    this.floatAnimation = this.animation.play('pickup_float', {
      target: this.mesh,
      update: (progress) => {
        this.mesh.position.y = baseY +
          Math.sin(progress * frequency * Math.PI * 2) * amplitude;
        this.mesh.rotation.y += 0.01;
      },
      loop: true
    });
  }

  onPlayerEnter(entity) {
    if (!this.active || entity.type !== 'player') {
      return;
    }

    this.collect();
  }

  async collect() {
    this.active = false;

    // Play collect sound
    this.audio.playSFX('pickup_' + this.item.pickup.itemType);

    // Apply effect
    await this.vfx.trigger(this.item.effects.collect.type, {
      target: this.mesh,
      duration: this.item.effects.collect.duration
    });

    // Give to player
    game.emit('player:pickup', {
      type: this.item.pickup.itemType,
      amount: this.item.pickup.amount
    });

    // Remove from scene
    this.scene.removeObject(this.mesh);
    this.scene.unregisterTrigger(this.item.id);

    // Respawn if enabled
    if (this.item.pickup.respawn) {
      setTimeout(() => this.respawn(),
        this.item.pickup.respawnTime * 1000
      );
    }
  }

  async respawn() {
    this.active = true;
    await this.spawn();
    this.vfx.trigger('respawn', { target: this.mesh });
  }
}
```

### Variations

- **Ammo**: Different item type, adds to weapon ammo
- **Key Item**: One-time pickup, unlocks specific door
- **Currency**: Coins, gems, adds to score/wallet
- **Quest Item**: Triggers quest update when collected

---

## Pattern: Pressure Plate

**What It Does**: A floor plate that triggers when stepped on.

**When to Use It**: Puzzles, traps, secret doors, environment interactions.

**How It Works**: Detects weight on the plate. Can be one-time, toggle, or hold (requires continuous pressure).

### Code Example

```javascript
// 1. Create pressure plate data
export const PRESSURE_PLATE_PUZZLE = {
  id: 'pressure_plate_puzzle',
  name: 'Puzzle Plate',
  type: 'pressure_plate',
  model: '/assets/models/pressure_plate.glb',
  position: { x: 0, y: 0.1, z: 0 },

  trigger: {
    shape: 'box',
    size: { x: 2, y: 0.5, z: 2 }
  },

  pressurePlate: {
    mode: 'hold',     // 'hold', 'toggle', 'onetime'
    targetAction: 'open_gate',
    requiredWeight: 1,  // Minimum entities to trigger
    resetDelay: 0.5     // Seconds before reset (hold mode)
  },

  effects: {
    onPressed: {
      animation: 'press_down',
      sound: 'plate_click',
      vfx: 'activate_glow'
    },
    onReleased: {
      animation: 'press_up',
      sound: 'plate_release'
    }
  }
};

// 2. Pressure plate controller
class PressurePlateController {
  constructor(plateData, sceneManager, audio, vfx) {
    this.plate = plateData;
    this.scene = sceneManager;
    this.audio = audio;
    this.vfx = vfx;

    this.isPressed = false;
    this.entitiesOnPlate = new Set();
    this.resetTimer = null;
  }

  initialize() {
    // Create plate mesh
    this.createMesh();

    // Register trigger
    this.scene.registerTrigger(this.plate.id, {
      shape: this.plate.trigger.shape,
      ...this.plate.trigger,
      onEnter: (entity) => this.onEntityEnter(entity),
      onExit: (entity) => this.onEntityExit(entity)
    });
  }

  createMesh() {
    // Load plate mesh with two parts: base and button
    this.mesh = {
      base: this.scene.loadModel(this.plate.model + '_base'),
      button: this.scene.loadModel(this.plate.model + '_button')
    };

    const pos = this.plate.position;
    this.mesh.base.position.set(pos.x, pos.y, pos.z);
    this.mesh.button.position.set(pos.x, pos.y + 0.05, pos.z);
  }

  onEntityEnter(entity) {
    // Check if entity has enough weight
    if (entity.weight < this.plate.pressurePlate.requiredWeight) {
      return;
    }

    this.entitiesOnPlate.add(entity.id);

    // Check if enough entities on plate
    if (this.entitiesOnPlate.size >= this.plate.pressurePlate.requiredWeight) {
      this.press();
    }
  }

  onEntityExit(entity) {
    this.entitiesOnPlate.delete(entity.id);

    // Check if still enough entities
    const hasWeight = this.entitiesOnPlate.size >=
      this.plate.pressurePlate.requiredWeight;

    if (!hasWeight && this.isPressed) {
      this.scheduleRelease();
    }
  }

  press() {
    if (this.isPressed &&
        this.plate.pressurePlate.mode === 'toggle') {
      // Toggle mode: release
      this.release();
      return;
    }

    if (this.isPressed) {
      return;  // Already pressed
    }

    this.isPressed = true;
    this.clearResetTimer();

    // Animate button down
    this.animateButton(true);

    // Play effects
    this.audio.playSFX(this.plate.effects.onPressed.sound);
    this.vfx.trigger(this.plate.effects.onPressed.vfx, {
      target: this.mesh.button
    });

    // Trigger action
    game.emit(this.plate.pressurePlate.targetAction, {
      source: this.plate.id,
      active: true
    });
  }

  scheduleRelease() {
    if (this.plate.pressurePlate.mode === 'onetime') {
      return;  // Stay pressed forever
    }

    if (this.plate.pressurePlate.mode === 'toggle') {
      return;  // Manual toggle only
    }

    // Hold mode: release after delay
    this.resetTimer = setTimeout(() => {
      if (!this.isPressed) return;

      const hasWeight = this.entitiesOnPlate.size >=
        this.plate.pressurePlate.requiredWeight;

      if (!hasWeight) {
        this.release();
      }
    }, this.plate.pressurePlate.resetDelay * 1000);
  }

  release() {
    if (!this.isPressed) return;

    this.isPressed = false;

    // Animate button up
    this.animateButton(false);

    // Play effects
    this.audio.playSFX(this.plate.effects.onReleased.sound);

    // Trigger action
    game.emit(this.plate.pressurePlate.targetAction, {
      source: this.plate.id,
      active: false
    });
  }

  animateButton(pressed) {
    const targetY = pressed ?
      this.plate.position.y - 0.03 :
      this.plate.position.y + 0.05;

    this.animation.play('plate_button', {
      target: this.mesh.button,
      position: { y: targetY },
      duration: 0.1,
      easing: 'easeOutQuad'
    });
  }

  clearResetTimer() {
    if (this.resetTimer) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
  }
}
```

### Variations

- **Weight-Sensitive**: Different effects based on how many entities
- **Timed**: Stays active for set time then releases
- **Combination Lock**: Multiple plates must be pressed together
- **Player-Only**: Only player's weight counts, not enemies/objects

---

## ENEMY PATTERNS

---

## Pattern: Patrolling Guard

**What It Does**: An enemy that walks between waypoints and investigates when player is spotted.

**When to Use It**: Basic enemy behavior in stealth/action games.

**How It Works**: The enemy follows a path of waypoints. When the player enters detection range, the enemy pauses and investigates.

### Code Example

```javascript
// 1. Create guard data
export const ENEMY_PATROL_GUARD = {
  id: 'guard_01',
  name: 'Patrol Guard',
  type: 'enemy',
  model: '/assets/models/guard.glb',
  position: { x: 0, y: 0, z: 0 },

  patrol: {
    waypoints: [
      { x: 0, y: 0, z: 0 },
      { x: 10, y: 0, z: 0 },
      { x: 10, y: 0, z: 10 },
      { x: 0, y: 0, z: 10 }
    ],
    loop: true,
    waitTime: 2,  // seconds at each waypoint
    moveSpeed: 2
  },

  detection: {
    sightRange: 15,
    sightAngle: Math.PI / 3,  // 60 degrees
    hearingRange: 5,
    investigationTime: 5
  },

  combat: {
    health: 100,
    damage: 10,
    attackRange: 2,
    attackCooldown: 1.5
  }
};

// 2. Guard AI controller
class PatrolGuardController {
  constructor(enemyData, sceneManager, physics, input) {
    this.enemy = enemyData;
    this.scene = sceneManager;
    this.physics = physics;
    this.input = input;

    this.state = 'patrol';
    this.currentWaypoint = 0;
    this.waitTimer = 0;
    this.lastKnownPlayerPos = null;
    this.attackTimer = 0;
  }

  update(deltaTime) {
    switch (this.state) {
      case 'patrol':
        this.updatePatrol(deltaTime);
        break;
      case 'investigate':
        this.updateInvestigate(deltaTime);
        break;
      case 'chase':
        this.updateChase(deltaTime);
        break;
      case 'attack':
        this.updateAttack(deltaTime);
        break;
    }

    // Always check for player
    this.checkDetection();
  }

  updatePatrol(deltaTime) {
    if (this.waitTimer > 0) {
      this.waitTimer -= deltaTime;
      return;
    }

    const target = this.enemy.patrol.waypoints[this.currentWaypoint];
    const reached = this.moveTo(target, deltaTime);

    if (reached) {
      // Wait at waypoint
      this.waitTimer = this.enemy.patrol.waitTime;

      // Next waypoint
      this.currentWaypoint =
        (this.currentWaypoint + 1) % this.enemy.patrol.waypoints.length;
    }
  }

  updateInvestigate(deltaTime) {
    if (!this.lastKnownPlayerPos) {
      this.state = 'patrol';
      return;
    }

    const reached = this.moveTo(this.lastKnownPlayerPos, deltaTime);

    if (reached) {
      // Look around briefly, then return to patrol
      this.waitTimer -= deltaTime;
      if (this.waitTimer <= 0) {
        this.lastKnownPlayerPos = null;
        this.state = 'patrol';
      }
    }
  }

  updateChase(deltaTime) {
    const playerPos = this.getPlayerPosition();
    this.moveTo(playerPos, deltaTime);

    // Check if in attack range
    const distance = this.getDistanceToPlayer();
    if (distance <= this.enemy.combat.attackRange) {
      this.state = 'attack';
    }
  }

  updateAttack(deltaTime) {
    this.attackTimer -= deltaTime;

    if (this.attackTimer <= 0) {
      this.performAttack();
      this.attackTimer = this.enemy.combat.attackCooldown;
    }

    // Check if player moved out of range
    const distance = this.getDistanceToPlayer();
    if (distance > this.enemy.combat.attackRange * 1.5) {
      this.state = 'chase';
    }
  }

  checkDetection() {
    const player = this.getPlayer();
    if (!player) return;

    const distance = this.getDistanceTo(player.position);
    const canSee = this.isInLineOfSight(player);

    if (distance <= this.enemy.detection.sightRange && canSee) {
      // Player spotted!
      this.lastKnownPlayerPos = player.position.clone();
      this.state = 'chase';

      // Alert!
      game.emit('enemy:spotted', {
        enemy: this.enemy.id,
        position: this.getPosition()
      });
    } else if (distance <= this.enemy.detection.hearingRange) {
      // Player heard, investigate
      this.lastKnownPlayerPos = player.position.clone();
      this.waitTimer = this.enemy.detection.investigationTime;
      this.state = 'investigate';
    }
  }

  moveTo(target, deltaTime) {
    const direction = new THREE.Vector3(
      target.x - this.position.x,
      0,
      target.z - this.position.z
    ).normalize();

    this.position.x += direction.x * this.enemy.patrol.moveSpeed * deltaTime;
    this.position.z += direction.z * this.enemy.patrol.moveSpeed * deltaTime;

    this.rotation.y = Math.atan2(direction.x, direction.z);

    // Check if reached
    const distance = Math.sqrt(
      Math.pow(target.x - this.position.x, 2) +
      Math.pow(target.z - this.position.z, 2)
    );

    return distance < 0.1;
  }

  isInLineOfSight(target) {
    // Raycast to check for obstacles
    const hit = this.physics.raycast(
      this.position,
      target.position.clone().sub(this.position).normalize(),
      this.enemy.detection.sightRange
    );

    return !hit || hit.entity === target;
  }

  performAttack() {
    // Play attack animation
    this.animation.play('attack', { duration: 0.5 });

    // Deal damage to player
    game.emit('enemy:attack', {
      enemy: this.enemy.id,
      damage: this.enemy.combat.damage
    });

    // Play attack sound
    game.getManager('audio').playSFX('guard_attack', {
      position: this.position
    });
  }

  getPlayerPosition() {
    const player = this.getPlayer();
    return player ? player.position : new THREE.Vector3();
  }

  getDistanceToPlayer() {
    return this.getDistanceTo(this.getPlayerPosition());
  }

  getDistanceTo(position) {
    return this.position.distanceTo(new THREE.Vector3(
      position.x, position.y, position.z
    ));
  }
}
```

### Variations

- **Stationary Guard**: Looks around but doesn't patrol
- **Chase-Only**: No patrol, chases on sight
- **Ranged Enemy**: Stops and shoots instead of melee attack
- **Fleeing**: Runs away from player instead of chasing

---

## UI PATTERNS

---

## Pattern: Dialogue Choice Consequence

**What It Does**: Dialog choices that have lasting consequences on the game world.

**When to Use It**: Branching narratives, reputation systems, quest outcomes.

**How It Works**: Choices are recorded and can be checked later to change dialog options, NPC behavior, or world state.

### Code Example

```javascript
// 1. Dialog with consequences (in DialogData.js)
export const DIALOG_MERCHANT_INTRO = {
  id: 'merchant_intro',
  startNode: 'greeting',

  nodes: {
    greeting: {
      speaker: 'merchant',
      text: 'Welcome, traveler! What brings you to my shop?',

      choices: [
        {
          text: "I'm looking to buy supplies.",
          nextNode: 'shopping',
          consequences: {
            mood: 'friendly',
            reputation: 5
          }
        },
        {
          text: 'Just browsing.',
          nextNode: 'browsing',
          consequences: {
            mood: 'neutral'
          }
        },
        {
          text: 'Your prices are too high!',
          condition: 'merchant:not_haggled',
          nextNode: 'haggle',
          consequences: {
            mood: 'annoyed',
            reputation: -5,
            flags: ['merchant:haggled']
          }
        }
      ]
    },

    shopping: {
      speaker: 'merchant',
      text: 'Excellent! I have fine wares for sale.',
      choices: [
        {
          text: 'Show me your wares.',
          action: 'open_shop',
          nextNode: null
        }
      ]
    },

    browsing: {
      speaker: 'merchant',
      text: 'Very well. Let me know if you need anything.',
      choices: [
        {
          text: 'Thanks.',
          nextNode: null
        }
      ]
    },

    haggle: {
      speaker: 'merchant',
      text: 'My prices are fair! But... perhaps we can work something out.',
      choices: [
        {
          text: 'Sorry, I spoke too hastily.',
          nextNode: 'shopping',
          consequences: {
            flags: ['merchant:apologized']
          }
        }
      ]
    }
  }
};

// 2. Consequence tracking system
class ConsequenceTracker {
  constructor() {
    this.flags = new Set();
    this.reputation = new Map();
    this.choices = [];
  }

  applyConsequences(consequences) {
    if (!consequences) return;

    // Add flags
    if (consequences.flags) {
      for (const flag of consequences.flags) {
        this.flags.add(flag);
      }
    }

    // Update reputation
    if (consequences.reputation) {
      const npc = consequences.npc || 'global';
      const current = this.reputation.get(npc) || 0;
      this.reputation.set(npc, current + consequences.reputation);
    }

    // Record choice
    this.choices.push({
      timestamp: Date.now(),
      ...consequences
    });

    // Emit event for other systems
    game.emit('consequence:applied', consequences);
  }

  hasFlag(flag) {
    return this.flags.has(flag);
  }

  getReputation(npc) {
    return this.reputation.get(npc) || 0;
  }

  canChoose(dialogId, choiceIndex, dialogData) {
    const node = dialogData.nodes[dialogData.currentNode];
    const choice = node.choices[choiceIndex];

    // Check conditions
    if (choice.condition) {
      return this.evaluateCondition(choice.condition);
    }

    return true;
  }

  evaluateCondition(condition) {
    // Parse conditions like "merchant:not_haggled"
    const [npc, flag] = condition.split(':');
    const fullFlag = condition;

    if (flag === 'not_haggled') {
      return !this.hasFlag(fullFlag);
    }

    return this.hasFlag(fullFlag);
  }
}
```

### Variations

- **Reputation Gates**: Choices only available at certain reputation levels
- **Flag Combinations**: AND/OR conditions on multiple flags
- **Timed Choices**: Player must choose within time limit
- **Skill Checks**: Choices may succeed or fail based on player stats

---

## PLAYER PATTERNS

---

## Pattern: Checkpoint System

**What It Does**: Save player progress at specific points, allowing respawning on death.

**When to Use It**: Any game with hazards or combat where player can die.

**How It Works**: When player touches a checkpoint, it becomes active. On death, player respawns at the last active checkpoint.

### Code Example

```javascript
// 1. Checkpoint data
export const CHECKPOINT_LEVEL1_MID = {
  id: 'checkpoint_level1_mid',
  name: 'Midpoint',
  type: 'checkpoint',
  model: '/assets/models/checkpoint.glb',
  position: { x: 25, y: 0, z: 0 },

  trigger: {
    shape: 'box',
    size: { x: 3, y: 4, z: 3 }
  },

  checkpoint: {
    savePlayer: true,        // Save health, ammo
    saveWorld: true,         // Save doors, items
    respawnFacing: { x: 0, y: 0, z: 1 }
  },

  effects: {
    inactive: { color: 0x666666, particles: false },
    active: { color: 0x00ff00, particles: true },
    justActivated: {
      flash: true,
      flashDuration: 1.0,
      sound: 'checkpoint_activate'
    }
  }
};

// 2. Checkpoint controller
class CheckpointController {
  constructor(checkpointData, sceneManager, audio, vfx) {
    this.checkpoint = checkpointData;
    this.scene = sceneManager;
    this.audio = audio;
    this.vfx = vfx;

    this.isActive = false;
    this.mesh = null;
  }

  initialize() {
    this.createMesh();
    this.registerTrigger();
    this.loadState();
  }

  createMesh() {
    this.mesh = this.scene.loadModel(this.checkpoint.model);
    this.mesh.position.set(
      this.checkpoint.position.x,
      this.checkpoint.position.y,
      this.checkpoint.position.z
    );

    // Set initial appearance
    this.updateAppearance();
  }

  registerTrigger() {
    this.scene.registerTrigger(this.checkpoint.id, {
      shape: this.checkpoint.trigger.shape,
      ...this.checkpoint.trigger,
      onEnter: (entity) => this.onPlayerEnter(entity)
    });
  }

  onPlayerEnter(entity) {
    if (entity.type !== 'player') return;
    if (this.isActive) return;  // Already active

    this.activate();
  }

  activate() {
    this.isActive = true;
    this.updateAppearance();

    // Play activation effects
    this.audio.playSFX(this.checkpoint.effects.justActivated.sound, {
      position: this.checkpoint.position
    });

    this.vfx.trigger('flash', {
      target: this.mesh,
      color: this.checkpoint.effects.active.color,
      duration: this.checkpoint.effects.justActivated.flashDuration
    });

    // Save checkpoint state
    this.saveState();

    // Save player state
    if (this.checkpoint.checkpoint.savePlayer) {
      this.savePlayerState();
    }

    // Save world state
    if (this.checkpoint.checkpoint.saveWorld) {
      this.saveWorldState();
    }

    // Notify game
    game.emit('checkpoint:activated', {
      id: this.checkpoint.id,
      position: this.checkpoint.position
    });
  }

  updateAppearance() {
    const colors = this.isActive ?
      this.checkpoint.effects.active :
      this.checkpoint.effects.inactive;

    this.mesh.material.color.setHex(colors.color);

    // Enable/disable particles
    if (this.isActive && colors.particles) {
      this.vfx.trigger('checkpoint_particles', {
        target: this.mesh,
        loop: true
      });
    }
  }

  saveState() {
    game.getManager('save').setData('lastCheckpoint', {
      id: this.checkpoint.id,
      position: this.checkpoint.position,
      facing: this.checkpoint.checkpoint.respawnFacing,
      timestamp: Date.now()
    });
  }

  savePlayerState() {
    const player = game.getManager('player');

    game.getManager('save').setData('playerAtCheckpoint', {
      health: player.getHealth(),
      maxHealth: player.getMaxHealth(),
      ammo: player.getAmmo(),
      inventory: player.getInventory()
    });
  }

  saveWorldState() {
    // Save door states, item pickup states, etc.
    const worldState = {};

    // Doors
    for (const [id, door] of this.scene.getDoors()) {
      worldState['door:' + id] = {
        isOpen: door.isOpen
      };
    }

    // Items
    for (const [id, item] of this.scene.getItems()) {
      worldState['item:' + id] = {
        collected: item.isCollected
      };
    }

    game.getManager('save').setData('worldAtCheckpoint', worldState);
  }

  loadState() {
    const lastCheckpoint = game.getManager('save').getData('lastCheckpoint');

    if (lastCheckpoint && lastCheckpoint.id === this.checkpoint.id) {
      this.isActive = true;
      this.updateAppearance();
    }
  }

  static respawn() {
    const save = game.getManager('save');
    const lastCheckpoint = save.getData('lastCheckpoint');

    if (!lastCheckpoint) {
      // No checkpoint, use level start
      return { x: 0, y: 0, z: 0 };
    }

    // Restore player state
    const playerState = save.getData('playerAtCheckpoint');
    if (playerState) {
      const player = game.getManager('player');
      player.setHealth(playerState.health);
      player.setMaxHealth(playerState.maxHealth);
      player.setAmmo(playerState.ammo);
      player.setInventory(playerState.inventory);
    }

    // Restore world state
    const worldState = save.getData('worldAtCheckpoint');
    if (worldState) {
      for (const [key, state] of Object.entries(worldState)) {
        const [type, id] = key.split(':');
        game.emit('restore:' + type, { id, state });
      }
    }

    return lastCheckpoint.position;
  }
}

// 3. Use checkpoint on player death
game.on('player:died', () => {
  const respawnPos = CheckpointController.respawn();
  const player = game.getManager('player');
  player.setPosition(respawnPos);
  player.revive();
});
```

### Variations

- **One-Time Checkpoint**: Can only be activated once
- **Conditional Checkpoint**: Only activates if criteria met (enemies defeated, etc.)
- **Multiple Respawn Points**: Choose from several spawn points
- **Shared Checkpoints**: Last checkpoint active for all players (co-op)

---

## More Patterns

### Inventory System

**What It Does**: Manage items player carries.

```javascript
class Inventory {
  constructor(slots = 20) {
    this.slots = slots;
    this.items = new Map();
  }

  addItem(itemId, quantity = 1) {
    const current = this.items.get(itemId) || 0;
    this.items.set(itemId, current + quantity);
    game.emit('inventory:added', { itemId, quantity });
  }

  removeItem(itemId, quantity = 1) {
    const current = this.items.get(itemId) || 0;
    if (current < quantity) return false;

    this.items.set(itemId, current - quantity);
    if (this.items.get(itemId) === 0) {
      this.items.delete(itemId);
    }
    game.emit('inventory:removed', { itemId, quantity });
    return true;
  }

  hasItem(itemId, quantity = 1) {
    return (this.items.get(itemId) || 0) >= quantity;
  }

  getItemCount(itemId) {
    return this.items.get(itemId) || 0;
  }
}
```

---

### Health Pickup

**What It Does**: Restores player health when collected.

```javascript
game.on('player:pickup', (data) => {
  const player = game.getManager('player');

  if (data.type === 'health') {
    const currentHealth = player.getHealth();
    const maxHealth = player.getMaxHealth();
    const healAmount = Math.min(data.amount, maxHealth - currentHealth);

    if (healAmount > 0) {
      player.setHealth(currentHealth + healAmount);
      game.getManager('audio').playSFX('health_pickup');
      game.getManager('vfx').trigger('heal', {
        target: player.mesh,
        color: 0x00ff00
      });
    }
  }
});
```

---

### Save Game System

**What It Does**: Save and load game progress.

```javascript
class SaveManager {
  constructor() {
    this.saveSlots = 3;
    this.currentData = null;
  }

  save(slotIndex) {
    const saveData = {
      version: 1,
      timestamp: Date.now(),
      player: this.getPlayerData(),
      world: this.getWorldData(),
      quests: this.getQuestData(),
      flags: Array.from(game.dialog.flags)
    };

    localStorage.setItem(`save_${slotIndex}`, JSON.stringify(saveData));
    game.emit('game:saved', { slot: slotIndex });
  }

  load(slotIndex) {
    const data = localStorage.getItem(`save_${slotIndex}`);
    if (!data) return false;

    const saveData = JSON.parse(data);
    this.restorePlayer(saveData.player);
    this.restoreWorld(saveData.world);
    this.restoreQuests(saveData.quests);
    game.dialog.flags = new Set(saveData.flags);

    game.emit('game:loaded', { slot: slotIndex });
    return true;
  }

  getPlayerData() {
    const player = game.getManager('player');
    return {
      position: player.getPosition(),
      health: player.getHealth(),
      inventory: Array.from(player.inventory.entries())
    };
  }
}
```

---

## Quick Reference

```
COMMON PATTERNS SUMMARY:

Interactions:
  - Simple Door (open/close)
  - Pick-Up Item (collectibles)
  - Pressure Plate (puzzles)
  - Lever/Toggle (manual triggers)

Enemies:
  - Patrolling Guard (waypoints)
  - Stationary Sentry (detection only)
  - Chase Enemy (follows player)
  - Ranged Attacker (shoots from distance)

UI:
  - Dialog Choices (branching narrative)
  - HUD Updates (health, ammo display)
  - Menu Navigation (settings, pause)
  - Notifications (objective updates)

Player:
  - Checkpoint System (respawn points)
  - Inventory Management (items)
  - Health/Ammo Pickup (resources)
  - Save/Load Game (persistence)
```

---

## Related Systems

- [Interactive Objects](../05-interactive-objects/interactive-object-system.md) - Object interaction
- [Dialog System](../06-narrative/dialog-system.md) - Conversations
- [Physics Manager](../04-input-physics/physics-manager.md) - Collisions and triggers
- [Save System](../09-save-system/save-system.md) - Persistence

---

## Source File Reference

**Pattern Implementations:**
- `src/patterns/DoorController.js` - Door behavior
- `src/patterns/PickupController.js` - Item collection
- `src/patterns/PressurePlateController.js` - Floor triggers
- `src/patterns/PatrolGuardController.js` - Enemy AI
- `src/patterns/CheckpointController.js` - Save points

**Utility:**
- `src/patterns/ConsequenceTracker.js` - Dialog consequences
- `src/patterns/Inventory.js` - Item management
- `src/patterns/SaveManager.js` - Save/load

---

## References

- [Game Programming Patterns](https://gameprogrammingpatterns.com/) - Design patterns
- [Behavior Trees](https://www.gamasutra.com/blogs/ChrisSimpson/20140717/223975/) - AI patterns
- [Unity Design Patterns](https://unity.com/) - Common game patterns

*Documentation last updated: January 12, 2026*
