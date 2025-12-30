/**
 * colliderData.js - TRIGGER ZONE DEFINITIONS
 * =============================================================================
 *
 * ROLE: Centralized definitions for all trigger colliders that detect player
 * entry/exit and set game state accordingly.
 *
 * COLLIDER TYPES:
 * - box: Axis-aligned box with half-extents
 * - sphere: Spherical trigger with radius
 * - capsule: Capsule shape with halfHeight and radius
 * - zone: Complex mesh-based zone (registered via SceneManager)
 *
 * PROPERTIES:
 * - id: Unique identifier
 * - type: Shape type
 * - position: {x, y, z} world position
 * - rotation: {x, y, z} in DEGREES
 * - dimensions: Shape-specific dimensions
 * - setStateOnEnter: State updates on player entry
 * - setStateOnExit: State updates on player exit
 * - once: If true, trigger only once
 * - criteria: State-based activation conditions
 * - blocking: If true, solid physics collider instead of trigger
 *
 * DESIGN NOTE:
 * Colliders should only set state - actual behaviors (dialog, music, animations)
 * are triggered by state changes in their respective managers.
 *
 * =============================================================================
 */

import { GAME_STATES } from "./gameData.js";
import { sceneObjects } from "./sceneData.js";

export const colliders = [
  {
    id: "trigger-phonebooth-ring",
    type: "box",
    position: sceneObjects.phonebooth.position,
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 10, y: 4, z: 10 },
    setStateOnEnter: { currentState: GAME_STATES.PHONE_BOOTH_RINGING },
    once: true, // Triggers once then cleans itself up
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
        $lt: GAME_STATES.PHONE_BOOTH_RINGING,
      },
    },
  },

  // Phonebooth interaction - only available after hearing the phone ring
  {
    id: "phonebooth-answer",
    type: "box",
    position: {
      x: sceneObjects.phonebooth.position.x,
      y: 1,
      z: sceneObjects.phonebooth.position.z,
    },
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 1, y: 4, z: 1 },
    setStateOnEnter: { currentState: GAME_STATES.ANSWERED_PHONE },
    once: true,
    enabled: true,
    criteria: { currentState: GAME_STATES.PHONE_BOOTH_RINGING }, // Only activates after phone starts ringing
  },

  {
    id: "cat",
    type: "box",
    position: { x: -0.5, y: 0.4, z: 18.6 },
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 2.5, y: 1.0, z: 2.5 },
    setStateOnEnter: { heardCat: true },
    true: false,
    enabled: true,
  },

  // Radio proximity trigger (one-time state progression)
  {
    id: "radio-state-trigger",
    type: "sphere",
    position: sceneObjects.radio.position,
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { radius: 8 },
    setStateOnEnter: { currentState: GAME_STATES.NEAR_RADIO },
    once: true, // Only trigger once - state progresses forward only
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
        $lt: GAME_STATES.NEAR_RADIO,
      },
    },
  },

  // Radio proximity toggle (for music crossfade)
  {
    id: "radio-proximity-toggle",
    type: "sphere",
    position: sceneObjects.radio.position,
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { radius: 8 },
    setStateOnEnter: { nearRadio: true },
    setStateOnExit: { nearRadio: false },
    once: false, // Repeatable - toggles on/off
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
      },
    },
  },

  // Shadow glimpse trigger
  {
    id: "shadow-glimpse-trigger",
    type: "sphere",
    position: { x: -10.45, y: 1.21, z: 40.64 },
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { radius: 3 },
    setStateOnEnter: { shadowGlimpse: true },
    once: true, // Only trigger once
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
      },
    },
  },

  // ENTERING_OFFICE trigger
  {
    id: "entering-office",
    type: "box",
    position: { x: -1.3, y: 1.33, z: 81.18 },
    rotation: { x: 0.0, y: 0.138, z: 0.0 },
    dimensions: { x: 3, y: 4, z: 3 },
    setStateOnEnter: { currentState: GAME_STATES.DOORS_CLOSE },
    once: true, // Only trigger once
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.DOORS_CLOSE,
      },
    },
  },

  // OFFICE_INTERIOR trigger
  {
    id: "interior-office",
    type: "box",
    position: sceneObjects.candlestickPhone.position,
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 3, y: 4, z: 3 },
    setStateOnEnter: { currentState: GAME_STATES.OFFICE_INTERIOR },
    once: true, // Only trigger once
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
  },

  // OFFICE_INTERIOR trigger
  {
    id: "candlestickPhone-pickup",
    type: "box",
    position: sceneObjects.candlestickPhone.position,
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 1.5, y: 5, z: 1.5 },
    setStateOnEnter: { currentState: GAME_STATES.OFFICE_PHONE_ANSWERED },
    once: true, // Only trigger once
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.OFFICE_INTERIOR,
        $lt: GAME_STATES.OFFICE_PHONE_ANSWERED,
      },
    },
  },

  // Blocking collider to prevent player backtracking after entering office
  {
    id: "doors-blocking",
    type: "box",
    position: { x: 5.68, y: 1.95, z: 78.7 }, // Same position as doors object
    rotation: { x: 180, y: 62, z: 180 }, // Converted from radians (3.1416, 1.0817, 3.1416)
    dimensions: { x: 2, y: 1, z: 0.1 }, // Half-extents: 1m wide, 2m tall, 0.2m thick
    blocking: true, // Solid blocking collider (prevents movement)
    enabled: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.ENTERING_OFFICE,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
    },
  },

  // {
  //   id: "shoulderTap",
  //   type: "box",
  //   position: { x: 0, y: 1, z: 5 },
  //   rotation: { x: 0, y: 0, z: 0 },
  //   dimensions: { x: 5, y: 3, z: 2 },
  //   onEnter: [
  //     {
  //       type: "state",
  //       data: { key: "currentState", value: GAME_STATES.SHOULDER_TAP },
  //     },
  //   ],
  //   onExit: [],
  // },

  // Zone colliders for exterior splat loading/unloading
  // Note: These should match your trimesh colliders (ZoneCollider-*) from Blender.
  // If using trimesh colliders, you may need to approximate with box colliders here,
  // or extend ColliderManager to support trimesh triggers.
  // Positions/dimensions below are placeholders - update to match your Blender geometry.
  // Each collider sets currentZone state on enter/exit to control splat loading.
  {
    id: "zone-alleyIntro",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-AlleyIntro trimesh (trimesh colliders from GLTF are primary)
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "alleyIntro" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false, // Allow entering/exiting multiple times
    enabled: false, // Disabled - using trimesh colliders from ExteriorZoneColliders.glb instead
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  {
    id: "zone-alleyNavigable",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-AlleyNavigable trimesh (trimesh colliders from GLTF are primary)
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "alleyNavigable" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false, // Allow entering/exiting multiple times
    enabled: false, // Disabled - using trimesh colliders from ExteriorZoneColliders.glb instead
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  {
    id: "zone-fourWay",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-FourWay trimesh
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "fourWay" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false,
    enabled: true,
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  {
    id: "zone-threeWay",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-ThreeWay trimesh
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "threeWay" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false,
    enabled: true,
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  {
    id: "zone-threeWay2",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-ThreeWay2 trimesh
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "threeWay2" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false,
    enabled: true,
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  {
    id: "zone-plaza",
    type: "box", // TODO: Update position/dimensions to match ZoneCollider-Plaza trimesh
    position: { x: 0, y: 1, z: 0 }, // Placeholder - update with actual position
    rotation: { x: 0, y: 0, z: 0 }, // Placeholder - update with actual rotation
    dimensions: { x: 10, y: 4, z: 10 }, // Placeholder - update with actual dimensions
    setStateOnEnter: { currentZone: "plaza" },
    setStateOnExit: { currentZone: null }, // Clear zone when exiting
    once: false,
    enabled: true,
    criteria: {
      currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // Only active before office
    },
  },

  // Add your colliders here...
];

export default colliders;
