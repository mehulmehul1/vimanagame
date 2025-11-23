/**
 * Scene Data Structure
 *
 * Defines scene objects like splat meshes and GLTF models.
 *
 * Each object contains:
 * - id: Unique identifier
 * - type: Type of object ('splat', 'gltf', etc.)
 * - path: Path to the asset file
 * - position: {x, y, z} position in 3D space
 * - rotation: {x, y, z} rotation in radians (Euler angles)
 * - scale: {x, y, z} scale multipliers or uniform number
 * - description: Human-readable description
 * - options: Type-specific options
 *   - useContainer: (GLTF) Wrap model in a container group
 *   - visible: If false, object will be invisible (useful for animation helpers)
 *   - physicsCollider: (GLTF) If true, create a Rapier trimesh collider from mesh geometry
 *   - debugMaterial: (GLTF) If true, apply semi-transparent wireframe material for debugging
 *   - envMap: (GLTF) Render environment map from splat scene and apply to materials
 *     - Note: World center position comes from the splat's envMapWorldCenter property
 *     - hideObjects: Array of object IDs to hide when rendering env map (default: [this object])
 *     - metalness: Default material metalness 0-1 (default: 1.0)
 *     - roughness: Default material roughness 0-1 (default: 0.02)
 *     - envMapIntensity: Default intensity/strength of reflection 0-5+ (default: 1.0)
 *     - materials: Array of material names to apply to (default: all materials)
 *     - excludeMaterials: Array of material names to skip (e.g., ["wood"])
 *     - materialOverrides: Object with per-material settings { materialName: { metalness, roughness, envMapIntensity } }
 *   - materialRenderOrder: (GLTF) Set renderOrder on meshes with specific materials
 *     - Object mapping material names to config: { materialName: { renderOrder, criteria } }
 *     - renderOrder: THREE.js renderOrder value (higher = rendered later)
 *     - criteria: Optional state criteria for conditional application
 *     - Example: { Vignette: { renderOrder: 9999, criteria: { currentState: { $gte: GAME_STATES.POST_DRIVE_BY } } } }
 * - envMapWorldCenter: (Splat) {x, y, z} position to render environment map from for this scene
 *   - contactShadow: (GLTF) Create contact shadows under the object using depth rendering
 *     - size: {x, y} - Plane dimensions (default: {x: 0.5, y: 0.5})
 *     - offset: {x, y, z} - Position offset relative to object (default: {x: 0, y: -0.05, z: 0})
 *     - blur: Shadow blur amount (default: 3.5)
 *     - darkness: Shadow darkness multiplier (default: 1.5)
 *     - opacity: Overall shadow opacity 0-1 (default: 0.5)
 *     - cameraHeight: Height of orthographic camera for shadow (default: 0.25)
 *     - renderTargetSize: Shadow texture resolution (default: 512)
 *     - trackMesh: Name of child mesh to track (for animated models, e.g. "Old_Car_01")
 *     - updateFrequency: Update every N frames (default: 3, lower = smoother/slower, higher = faster)
 *     - isStatic: If true, render once and never update again (best for objects that never move)
 *     - debug: Show camera helper for debugging (default: false)
 *     - criteria: Optional state criteria for conditional enable/disable (see note below)
 *     - Note: Contact shadows are automatically disabled when dissolve effects start (progress > -14.0).
 *       This happens in dissolveEffect.js during the transition, so shadows fade out before props dissolve.
 *       Don't add manual state-based enable/disable code elsewhere - the dissolve effect handles it.
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.CHAPTER_2 }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 * - loadByDefault: If true, load regardless of state (default: false)
 * - priority: Higher priority objects are loaded first (default: 0)
 * - zone: (Splat) Zone name for automatic zone-based loading/unloading (e.g., "fourWay", "threeWay2")
 *   - Objects with a zone property are automatically discovered by ZoneManager and loaded/unloaded with their zone
 *   - This eliminates the need to manually add assets to zone mapping arrays
 * - gizmo: If true, enable debug gizmo for positioning visual objects (G=move, R=rotate, S=scale)
 * - animations: Array of animation definitions (for GLTF objects with animations)
 *   - id: Unique identifier for this animation
 *   - clipName: Name of animation clip in GLTF (null = use first clip)
 *   - loop: Whether to loop the animation
 *   - criteria: Optional object with key-value pairs that must match game state for animation to play
 *     - Simple equality: { currentState: GAME_STATES.ANSWERED_PHONE }
 *     - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *     - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 *     - If criteria matches and not playing → play it
 *     - If criteria doesn't match and playing → stop it
 *     - Non-looping animations (loop: false) will not restart once finished
 *   - autoPlay: If true, automatically play when criteria are met
 *   - timeScale: Playback speed (1.0 = normal)
 *   - removeObjectOnFinish: If true, remove parent object from scene when animation finishes
 *   - onComplete: Optional callback when animation completes. Receives gameManager as parameter.
 *     Example: onComplete: (gameManager) => { gameManager.setState({...}); }
 *
 * Usage:
 * import { sceneObjects, getSceneObjectsForState } from './sceneData.js';
 *
 * Console Commands:
 * - List all available envMapWorldCenter positions:
 *   window.getEnvMapWorldCenters()
 *
 * - Generate environment map (for in-engine use):
 *   const envMap = await window.sceneManager.captureEnvMap({
 *     position: {x: -5.32, y: 2.5, z: 87.95},
 *     hideObjectIds: ['candlestickPhone', 'edison'],
 *     download: false
 *   })
 *
 * - For Blender IBL (manual screenshot approach):
 *   1. window.sceneManager.moveCameraToEnvMapCenter('interior')
 *   2. Take screenshot (F12 > Console > right-click canvas > Save image)
 *   3. Use screenshot as environment map in Blender
 */

import { GAME_STATES } from "./gameData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { Logger } from "./utils/logger.js";

// Create module-level logger
const logger = new Logger("SceneData", false);

const originPosition = { x: 0, y: 0, z: 0 };
const originRotation = { x: 0, y: 0, z: 0 };

const interiorPosition = { x: 5.36, y: 0.83, z: 78.39 };
const interiorRotation = { x: -3.1416, y: 1.0358, z: -3.1416 };

export const sceneObjects = {
  plaza: {
    id: "plaza",
    type: "splat",
    path: "/splats/plaza_16m.sog",
    description: "Plaza section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
  },

  plazaLaptop: {
    id: "plaza",
    type: "splat",
    path: "/splats/35/plaza_35.sog",
    description: "Plaza section of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
  },

  plazaDesktop: {
    id: "plaza",
    type: "splat",
    path: "/splats/8m/plaza_8m.sog",
    description: "Plaza section of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
  },

  plazaMobile: {
    id: "plaza",
    type: "splat",
    path: "/splats/1m/plaza-1m.sog",
    description: "Plaza section of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
  },

  fourWay: {
    id: "fourWay",
    type: "splat",
    path: "/splats/FourWay.sog",
    description:
      "Four-way intersection section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
  },

  fourWayLaptop: {
    id: "fourWay",
    type: "splat",
    path: "/splats/35/fourWay_35.sog",
    description:
      "Four-way intersection section of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
  },

  fourWayDesktop: {
    id: "fourWay",
    type: "splat",
    path: "/splats/8m/fourWay_8m.sog",
    description:
      "Four-way intersection section of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
  },

  fourWayMobile: {
    id: "fourWay",
    type: "splat",
    path: "/splats/1m/fourWay-1m.sog",
    description:
      "Four-way intersection section of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
  },

  alleyIntro: {
    id: "alleyIntro",
    type: "splat",
    path: "/splats/AlleyIntro.sog",
    description: "Intro alley section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 70, // Load first of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
    preload: true,
  },

  alleyIntroLaptop: {
    id: "alleyIntro",
    type: "splat",
    path: "/splats/35/alleyIntro_35.sog",
    description:
      "Intro alley section of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 70, // Load first of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
    preload: true,
  },

  alleyIntroDesktop: {
    id: "alleyIntro",
    type: "splat",
    path: "/splats/8m/alleyIntro_8m.sog",
    description:
      "Intro alley section of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 70, // Load first of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
    preload: true,
  },

  alleyIntroMobile: {
    id: "alleyIntro",
    type: "splat",
    path: "/splats/1m/alleyIntro-1m.sog",
    description:
      "Intro alley section of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 70, // Load first of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
    preload: true,
  },

  alleyLongView: {
    id: "alleyLongView",
    type: "splat",
    path: "/splats/AlleyLongView.sog",
    description:
      "Long view alley section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
  },

  alleyLongViewLaptop: {
    id: "alleyLongView",
    type: "splat",
    path: "/splats/35/alleyLongView_35.sog",
    description:
      "Long view alley section of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
  },

  alleyLongViewDesktop: {
    id: "alleyLongView",
    type: "splat",
    path: "/splats/8m/alleyLongView_8m.sog",
    description:
      "Long view alley section of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
  },

  alleyLongViewMobile: {
    id: "alleyLongView",
    type: "splat",
    path: "/splats/1m/alleyLongView-1m.sog",
    description:
      "Long view alley section of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
  },

  alleyNavigable: {
    id: "alleyNavigable",
    type: "splat",
    path: "/splats/AlleyNavigable.sog",
    description:
      "Navigable alley section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 68, // Load third of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max", // Only load on max (merged with alleyIntro on mobile, not used on laptop)
    },
  },

  threeWay: {
    id: "threeWay",
    type: "splat",
    path: "/splats/ThreeWay.sog",
    description:
      "Three-way intersection section of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
  },

  threeWayLaptop: {
    id: "threeWay",
    type: "splat",
    path: "/splats/35/threeWay_35.sog",
    description:
      "Three-way intersection section of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
  },

  threeWayDesktop: {
    id: "threeWay",
    type: "splat",
    path: "/splats/8m/threeWay_8m.sog",
    description:
      "Three-way intersection section of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
  },

  threeWayMobile: {
    id: "threeWay",
    type: "splat",
    path: "/splats/1m/threeWay-1m.sog",
    description:
      "Three-way intersection section of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 69, // Load second of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
  },

  threeWay2: {
    id: "threeWay2",
    type: "splat",
    path: "/splats/ThreeWay2.sog",
    description:
      "Three-way intersection section 2 of exterior environment (max quality)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 68, // Load third of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "max",
    },
  },

  threeWay2Laptop: {
    id: "threeWay2",
    type: "splat",
    path: "/splats/35/threeWay2_35.sog",
    description:
      "Three-way intersection section 2 of exterior environment (laptop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 68, // Load third of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "laptop",
    },
  },

  threeWay2Desktop: {
    id: "threeWay2",
    type: "splat",
    path: "/splats/8m/threeWay2_8m.sog",
    description:
      "Three-way intersection section 2 of exterior environment (desktop optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 68, // Load third of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "desktop",
    },
  },

  threeWay2Mobile: {
    id: "threeWay2",
    type: "splat",
    path: "/splats/1m/threeWay2-1m.sog",
    description:
      "Three-way intersection section 2 of exterior environment (mobile optimized)",
    position: { x: 0.35, y: 1.0, z: 1.9 },
    rotation: { x: 0, y: Math.PI, z: Math.PI },
    scale: { x: 1, y: 1, z: 1 },
    priority: 68, // Load third of deferred exterior pieces
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
      performanceProfile: "mobile",
    },
  },

  exteriorCollider: {
    id: "exteriorCollider",
    type: "gltf",
    path: "/gltf/colliders/ExteriorColliderOrigin.glb",
    description: "Exterior physics collider mesh",
    position: originPosition,
    rotation: originRotation,
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: false, // Set to true to see debug material
      physicsCollider: true, // Flag to create physics trimesh collider
      debugMaterial: false, // Apply semi-transparent debug material to visualize collider
      shadowBlocker: true, // Render as depth-only blocker for contact shadows (after splats, before shadows)
    },
    priority: 90,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
    },
  },

  exteriorZoneColliders: {
    id: "exteriorZoneColliders",
    type: "gltf",
    path: "/gltf/colliders/ExteriorZoneColliders.glb",
    description: "Exterior zone trigger colliders for splat loading/unloading",
    position: originPosition,
    rotation: originRotation,
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: true, // Hidden by default
      triggerColliders: true, // Flag to create trigger colliders from child meshes
    },
    priority: 90,
    preload: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.ENTERING_OFFICE,
      },
    },
  },

  interior: {
    id: "interior",
    type: "splat",
    path: "/green-room-clean.sog",
    description:
      "Main interior office environment splat mesh (desktop/max quality)",
    position: interiorPosition,
    rotation: { x: 0.0, y: -1.283, z: -3.1416 },
    scale: { x: 1.0, y: 1.0, z: 1.0 },
    priority: 67, // Load fourth (after threeWay2)
    envMapWorldCenter: { x: -5.32, y: 2.5, z: 87.95 }, // Position to render environment map from
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
      performanceProfile: { $in: ["desktop", "max"] },
    },
    preload: false,
  },

  interiorLaptop: {
    id: "interior",
    type: "splat",
    path: "/green-room-1m.sog",
    description:
      "Main interior office environment splat mesh (laptop optimized)",
    position: interiorPosition,
    rotation: { x: 0.0, y: -1.283, z: -3.1416 },
    scale: { x: 1.0, y: 1.0, z: 1.0 },
    priority: 67, // Load fourth (after threeWay2)
    envMapWorldCenter: { x: -5.32, y: 2.5, z: 87.95 }, // Position to render environment map from
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
      performanceProfile: "laptop",
    },
    preload: false,
  },

  interiorMobile: {
    id: "interior",
    type: "splat",
    path: "/green-room-1m.sog",
    description:
      "Main interior office environment splat mesh (mobile optimized)",
    position: interiorPosition,
    rotation: { x: 0.0, y: -1.283, z: -3.1416 },
    scale: { x: 1.0, y: 1.0, z: 1.0 },
    priority: 67, // Load fourth (after threeWay2)
    envMapWorldCenter: { x: -5.32, y: 2.5, z: 87.95 }, // Position to render environment map from
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
      performanceProfile: "mobile",
    },
    preload: false,
  },

  officeHell: {
    id: "officeHell",
    type: "splat",
    path: "/office-hell-500.sog",
    description:
      "Hell version of office interior (morphs from interior-nan-2.sog during VIEWMASTER_HELL) - laptop/mobile optimized",
    position: { x: -2.36, y: 2.73, z: 84.04 },
    rotation: { x: -0.0, y: -1.3373, z: 3.1416 },
    scale: { x: 2.06, y: 2.06, z: 2.06 },
    priority: 66, // Load fifth (after interior)
    envMapWorldCenter: { x: -5.32, y: 2.5, z: 87.95 }, // Same environment map center as interior
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER_HELL,
        $lte: GAME_STATES.POST_VIEWMASTER,
      },
      performanceProfile: { $in: ["mobile", "laptop", "desktop", "max"] },
    },
    preload: false,
  },

  club: {
    id: "club",
    type: "splat",
    path: "/club.sog",
    description: "Club environment splat mesh",
    position: { x: -6.36, y: 2.73, z: 82.26 },
    rotation: { x: 0, y: -Math.PI / 2, z: -Math.PI },
    scale: { x: 1.1, y: 1.1, z: 1.1 },
    priority: 65, // Load sixth (after officeHell)
    criteria: {
      currentState: {
        $gte: GAME_STATES.LIGHTS_OUT,
      },
      performanceProfile: { $ne: "mobile" },
    },
    preload: false,
  },

  clubMobile: {
    id: "club",
    type: "splat",
    path: "/club-1m.sog",
    gizmo: false,
    description: "Club environment splat mesh (mobile optimized)",
    position: { x: -3.6, y: 0.5, z: 85.29 },

    rotation: { x: 0.04, y: Math.PI / 2, z: -Math.PI },
    scale: { x: 1.9, y: 1.9, z: 1.9 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.LIGHTS_OUT,
      },
      performanceProfile: "mobile",
    },
    preload: false,
  },

  officeCollider: {
    id: "officeCollider",
    type: "gltf",
    path: "/gltf/colliders/office-collider-v01.glb",
    description: "Office interior physics collider mesh",
    position: originPosition,
    rotation: originRotation,
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: false, // Set to true to see debug material
      physicsCollider: true, // Flag to create physics trimesh collider
      debugMaterial: false, // Apply semi-transparent debug material to visualize collider
    },
    priority: 90,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
    preload: false,
  },

  phonebooth: {
    id: "phonebooth",
    type: "gltf",
    path: "/gltf/phonebooth.glb",
    description:
      "Phone booth GLTF model with CordAttach and Receiver children (uses PhoneCord module)",
    position: { x: 5.94, y: 0.27, z: 65.76 },
    rotation: { x: 0.0, y: Math.PI / 1.7, z: 0.0 },
    scale: { x: 1.5, y: 1.575, z: 1.5 },
    preload: true,
    options: {
      // Create a container group for proper scaling
      useContainer: true,
      contactShadow: {
        size: { x: 1, y: 1 }, // Capture area size
        offset: { x: 0, y: -0.01, z: 0 }, // Position offset
        blur: 2.5, // Slightly more blur for larger shadow
        darkness: 25.5, // Shadow darkness multiplier
        opacity: 0.9, // Overall shadow opacity
        cameraHeight: 2, // Taller camera for car
        shadowScale: { x: 1.15, y: 1.15 }, // Scale shadow plane display (extends beyond capture area)
        static: true,
      },
    },
    priority: 50,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
    animations: [
      {
        id: "phonebooth-ring",
        clipName: null,
        loop: false,
        criteria: {
          currentState: {
            $gte: GAME_STATES.ANSWERED_PHONE,
            $lt: GAME_STATES.DIALOG_CHOICE_1,
          },
        },
        autoPlay: true,
        timeScale: 1.0,
      },
    ],
  },

  radio: {
    id: "radio",
    type: "gltf",
    path: "/gltf/radio-1.glb",
    description: "Newsman's radio in the alley",
    position: { x: 4.35, y: 0.82, z: 37.29 },
    rotation: { x: -3.0991, y: 1.1719, z: -3.0852 },
    scale: { x: 2.46, y: 2.46, z: 2.46 },
    options: {
      useContainer: true,
      contactShadow: {
        size: { x: 0.5, y: 0.5 },
        offset: { x: 0, y: 0, z: 0 },
        shadowScale: { x: 1.15, y: 1.15 },
        blur: 2.5,
        darkness: 10.5,
        opacity: 0.9,
        cameraHeight: 0.25,
      },
    },
    priority: 50,
    preload: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
  },

  car: {
    id: "car",
    type: "gltf",
    preload: false,
    path: "/gltf/Old_Car_01.glb",
    description: "Animated car in drive-by shooting in plaza",
    position: { x: -15.67, y: 0.2, z: 62.5 },
    rotation: { x: 0.0, y: 0.8859, z: 0.0 },
    scale: { x: 0.9, y: 0.9, z: 0.9 },
    options: {
      useContainer: true,
      contactShadow: {
        size: { x: 2.5, y: 5.25 }, // Larger plane for car
        offset: { x: 0, y: 0.0, z: 0 }, // Position offset
        blur: 1.25, // Slightly more blur for larger shadow
        darkness: 1.0, // Shadow darkness multiplier
        opacity: 0.4, // Overall shadow opacity
        cameraHeight: 2.5, // Taller camera for car
        trackMesh: "Old_Car_01", // Track the actual car mesh (for animated models)
        updateFrequency: 1, // Update every frame (animated object)
        debug: false,
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.DRIVE_BY_PREAMBLE,
        $lt: GAME_STATES.POST_DRIVE_BY,
      },
    },
    priority: 50,
    animations: [
      {
        id: "drive-by-anim",
        clipName: null,
        loop: false,
        autoPlay: true,
        timeScale: 0.45,
        removeObjectOnFinish: true, // Despawn car after animation completes
        criteria: {
          currentState: {
            $gte: GAME_STATES.DRIVE_BY_PREAMBLE,
            $lt: GAME_STATES.POST_DRIVE_BY,
          },
        },
      },
    ],
  },

  doors: {
    id: "doors",
    type: "gltf",
    path: "/gltf/Basement_Door.glb",
    description: "Doors that lead to scene 2 interior",
    preload: false,

    position: { x: 5.68, y: 1.95, z: 78.7 },
    rotation: { x: 3.1416, y: 1.0817, z: 3.1416 },
    scale: { x: 0.32, y: 0.29, z: 0.38 },
    options: {
      useContainer: true,
    },
    criteria: {
      currentState: {
        $lt: GAME_STATES.LIGHTS_OUT,
      },
    },
    priority: 100,
  },

  rustedCar: {
    id: "rustedCar",
    type: "splat",
    path: "/rusted-car-2.sog",
    preload: false,
    zone: "fourWay",
    description: "Rusted car blocking the alley in fourWay zone",
    position: { x: 14.06, y: 0.91, z: 35.28 },
    rotation: { x: 0.0937, y: -1.048, z: -3.0978 },
    scale: { x: 3.65, y: 2.13, z: 2.13 },
    priority: 100,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
      performanceProfile: { $ne: "mobile" },
    },
  },

  train: {
    id: "train",
    type: "splat",
    path: "/train-bw.sog",
    zone: "threeWay2",
    description: "Train splat",
    position: { x: -40.61, y: -0.24, z: 67.04 },
    rotation: { x: -0.1179, y: 1.0486, z: -2.983 },
    scale: { x: 7.1, y: 7.1, z: 7.1 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
      performanceProfile: { $ne: "mobile" },
    },
  },

  coneCurve: {
    id: "coneCurve",
    type: "gltf",
    path: "/gltf/ConeCurve.glb",
    description: "Camera curve animation for start screen",
    position: { x: 0, y: 0, z: -0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: false, // Hide the cone - it's just a helper for camera animation
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
      },
    },
    priority: 100,
    animations: [
      {
        id: "coneCurve-anim",
        clipName: null, // Use first animation clip
        loop: true,
        autoPlay: true,
        timeScale: 0.15,
        criteria: {
          currentState: {
            $gte: GAME_STATES.START_SCREEN,
            $lt: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
          },
        },
      },
    ],
    preload: true,
  },

  viewmaster: {
    id: "viewmaster",
    type: "gltf",
    path: "/gltf/viewmaster.glb",
    preload: false,
    description: "Viewmaster GLTF model",
    position: { x: -8.24, y: 2.48, z: 85.39 },
    rotation: { x: -1.4084, y: 1.2006, z: 1.3865 },
    scale: { x: 1.0, y: 1.0, z: 1.0 },
    options: {
      contactShadow: {
        size: { x: 0.9, y: 0.9 }, // Plane dimensions
        offset: { x: 0, y: -0.275, z: 0 },
        blur: 3.5,
        darkness: 1.5,
        opacity: 0.8,
        cameraHeight: 0.5,
        updateFrequency: 5, // Interactive object - update more frequently
        debug: false,
        criteria: {
          // Don't want the state
          currentState: { $lte: GAME_STATES.PRE_VIEWMASTER },
        },
      },
      envMap: {
        // Render environment map from splat scene and apply reflections
        //metalness: 0.85, // Metallic for vintage brass/metal look
        //roughness: 0.15, // Some roughness for aged metal surface
        envMapIntensity: 0.75, // Boosted for visibility
        // hideObjects defaults to [this object]
      },
      materialRenderOrder: {
        Vignette: {
          renderOrder: 10000,
          criteria: {
            currentState: {
              $gte: GAME_STATES.VIEWMASTER_COLOR,
              $lte: GAME_STATES.POST_VIEWMASTER,
            },
          },
        },
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  amplifier: {
    id: "amplifier",
    type: "gltf",
    path: "/gltf/amplifier.glb",
    preload: false,
    description:
      "Amplifier GLTF model positioned next to viewmaster during LIGHTS_OUT",
    position: { x: -2.28, y: 0.86, z: 81.04 },
    rotation: { x: 0.0, y: -0.5129, z: 0.0 },
    scale: { x: 1.91, y: 2.04, z: 1.53 },
    options: {
      useContainer: true,
      contactShadow: {
        size: { x: 0.75, y: 0.75 }, // Plane dimensions
        offset: { x: 0, y: 0, z: 0 }, // Position offset
        blur: 1.5, // Shadow blur amount
        darkness: 1.5, // Shadow darkness multiplier
        opacity: 0.85, // Overall shadow opacity
        cameraHeight: 1.35, // Height for shadow camera
        isStatic: true, // Never moves - render once
        fadeDuration: 2.0, // Fade in/out duration in seconds
        debug: false, // Set to true to visualize the shadow camera AND see texture
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.LIGHTS_OUT,
      },
    },
    priority: 50,
  },

  edison: {
    id: "edison",
    type: "gltf",
    path: "/gltf/edison.glb",
    preload: false,
    description: "edison cylinder player",
    position: { x: -5.3, y: 2.11, z: 86.97 },
    rotation: { x: 3.1416, y: -0.0245, z: 3.1416 },
    scale: { x: 1.32, y: 1.32, z: 1.32 },
    options: {
      contactShadow: {
        size: { x: 0.75, y: 0.75 }, // Plane dimensions
        offset: { x: 0, y: 0, z: 0 }, // Position offset
        blur: 1.5, // Shadow blur amount
        darkness: 1.5, // Shadow darkness multiplier
        opacity: 0.85, // Overall shadow opacity
        cameraHeight: 1.35, // Height for shadow camera
        isStatic: false, // Never moves - render once
        fadeDuration: 2.0, // Fade in/out duration in seconds
        debug: false, // Set to true to visualize the shadow camera AND see texture
      },
      envMap: {
        // Render environment map from splat scene and apply reflections
        metalness: 0.85, // Metallic for vintage brass/metal look
        roughness: 0.15, // Some roughness for aged metal surface
        envMapIntensity: 1.0, // Boosted for visibility
        excludeMaterials: ["wood", "03 - frame"], // Don't apply to wood
        materialOverrides: {
          // Make deepbluepaint extremely reflective
          deepbluepaint: {
            metalness: 1.0,
            roughness: 0.1,
            envMapIntensity: 1.0,
          },
        },
        // hideObjects defaults to [this object]
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
    },
    animations: [
      {
        id: "edison-anim",
        clipName: null,
        loop: false,
        autoPlay: true,
        timeScale: 0.5,
        delay: 1.5,
        criteria: {
          currentState: GAME_STATES.EDISON,
        },
      },
    ],
  },

  candlestickPhone: {
    id: "candlestickPhone",
    type: "gltf",
    path: "/gltf/candlestickPhone.glb",
    preload: false,
    description:
      "Candlestick phone with CordAttach and Receiver children (uses PhoneCord module)",
    position: { x: -5.24, y: 2.13, z: 82.47 },
    rotation: { x: 3.1416, y: 2.0904, z: 3.1416 },
    scale: { x: 1.32, y: 1.32, z: 1.32 },
    options: {
      useContainer: true, // Wrap in container to preserve original model structure
      contactShadow: {
        size: { x: 0.4, y: 0.4 }, // Plane dimensions
        offset: { x: 0, y: -0.01, z: 0 }, // Position offset
        blur: 3.5, // Shadow blur amount
        darkness: 20.0, // Shadow darkness multiplier
        opacity: 0.7, // Overall shadow opacity
        cameraHeight: 0.25, // Height for shadow camera
        isStatic: true, // Never moves - render once
        fadeDuration: 2.0, // Fade in/out duration in seconds
        debug: false, // Set to true to visualize the shadow camera
      },
      envMap: {
        // Render environment map from splat scene and apply reflections
        envMapIntensity: 0.5, // Boosted for visibility
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
    },
    // Note: This object has a companion script (src/content/candlestickPhone.js)
    // The script manages the CordAttach and Receiver children using the PhoneCord module
    // Initialize in main.js or scene manager after the object is loaded
  },

  paSpeaker: {
    id: "paSpeaker",
    type: "gltf",
    path: "/gltf/pa.glb",
    description: "Podium for outro sequence",
    position: { x: -6.12, y: 4.21, z: 75.86 },
    rotation: { x: -2.8628, y: 0.979, z: -3.0555 },
    scale: { x: 2.14, y: 2.14, z: 2.14 },
    options: {
      useContainer: true,
    },
    priority: 100,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LIGHTS_OUT,
      },
    },
  },

  projectorScreen: {
    id: "projectorScreen",
    type: "gltf",
    path: "/gltf/projector-screen.glb",
    description: "Projector screen for outro sequence",
    position: { x: -0.24, y: 1.79, z: 83.33 },
    rotation: { x: -3.1416, y: -0.0078, z: -3.1416 },
    scale: { x: 0.81, y: 0.51, z: 0.51 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.OUTRO_CZAR,
      },
    },
  },

  letter: {
    id: "letter",
    type: "gltf",
    path: "/gltf/letter.glb",
    description: "Animated letter plane for outro sequence",
    position: { x: -4.28, y: 1.94, z: 78.77 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
    },
    priority: 500,
    preload: false,
    criteria: {
      currentState: {
        $in: [
          GAME_STATES.OUTRO,
          GAME_STATES.OUTRO_LECLAIRE,
          GAME_STATES.OUTRO_CAT,
        ],
      },
    },
    animations: [
      {
        id: "letter-anim",
        clipName: null, // Use first animation clip
        loop: false,
        autoPlay: true,
        timeScale: 0.5,
        criteria: {
          currentState: {
            $in: [
              GAME_STATES.OUTRO,
              GAME_STATES.OUTRO_LECLAIRE,
              GAME_STATES.OUTRO_CAT,
            ],
          },
        },
        onComplete: (gameManager) => {
          console.log("letter anim complete");
          gameManager.setState({ currentState: GAME_STATES.OUTRO_LECLAIRE });
        },
      },
    ],
  },

  //  A failed attempt at a first person body model positioned below the camera
  //  It worked okay but the state management and all the animation I'd need to do was onerous
  //  for my schedule. Documenting for posterity:

  //   firstPersonBody: {
  //     id: "firstPersonBody",
  //     type: "gltf",
  //     path: "/gltf/mixamo-test-rig.glb",
  //     description: "First-person body model (will be reparented to camera)",
  //     position: { x: 0, y: 0, z: 0 }, // Will be repositioned when attached to camera
  //     rotation: { x: 0, y: 0, z: 0 }, // Will be reoriented when attached to camera
  //     scale: 1.0,
  //     options: {
  //       useContainer: true,
  //     },
  //     loadByDefault: true,
  //     priority: 90, // Load early, before most objects but after exterior
  //     animations: [
  //       {
  //         id: "firstPersonBody-walk",
  //         clipName: "Walking (1)_3", // The walking animation
  //         loop: true,
  //         autoPlay: false, // CharacterController will manage this manually
  //         timeScale: 1.0,
  //       },
  //       {
  //         id: "firstPersonBody-idle",
  //         clipName: "Idle_1", // The idle animation
  //         loop: true,
  //         autoPlay: false, // CharacterController will manage this manually
  //         timeScale: 1.0,
  //       },
  // //     ],
  //   },
};

/**
 * Get scene objects that should be loaded for the current game state
 * @param {Object} gameState - Current game state
 * @param {Object} options - Options object
 * @param {boolean} options.preloadOnly - If true, only return objects with preload: true
 * @param {boolean} options.deferredOnly - If true, only return objects with preload: false
 * @param {boolean} options.forcePreloadForState - If true (debug mode), include all matching objects regardless of preload flag
 * @returns {Array<Object>} Array of scene objects that should be loaded
 */
export function getSceneObjectsForState(gameState, options = {}) {
  // Convert to array and sort by priority (descending)
  const sortedObjects = Object.values(sceneObjects).sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );

  const matchingObjects = [];

  for (const obj of sortedObjects) {
    // Filter by preload if requested (unless forcePreloadForState is true)
    // Treat undefined preload as false (deferred)
    const objPreload = obj.preload !== undefined ? obj.preload : false;

    // In debug mode with forcePreloadForState, skip preload filtering - include all matching objects
    if (!options.forcePreloadForState) {
      if (options.preloadOnly && objPreload !== true) {
        if (obj.id === "exteriorZoneColliders") {
          logger.log(
            `exteriorZoneColliders skipped: preloadOnly=true but preload=${objPreload}`
          );
        }
        continue;
      }
      if (options.deferredOnly && objPreload !== false) {
        if (obj.id === "exteriorZoneColliders") {
          logger.log(
            `exteriorZoneColliders skipped: deferredOnly=true but preload=${objPreload}`
          );
        }
        continue;
      }
    }

    // Always include objects marked as loadByDefault
    if (obj.loadByDefault === true) {
      matchingObjects.push(obj);
      continue;
    }

    // Check criteria (supports operators like $gte, $lt, etc.)
    if (obj.criteria) {
      const matches = checkCriteria(gameState, obj.criteria);
      if (!matches) {
        if (obj.id === "exteriorZoneColliders") {
          logger.log(
            `exteriorZoneColliders does NOT match criteria (state=${gameState.currentState})`,
            obj.criteria
          );
        }
        continue;
      }
      if (obj.id === "exteriorZoneColliders") {
        logger.log(
          `exteriorZoneColliders MATCHES criteria (state=${gameState.currentState})`
        );
      }
    }

    // If we get here, all conditions passed
    matchingObjects.push(obj);
    if (obj.id === "exteriorZoneColliders") {
      logger.log(`exteriorZoneColliders added to matchingObjects`);
    }
  }

  return matchingObjects;
}

/**
 * Get a scene object by ID
 * @param {string} id - Object ID
 * @returns {Object|null} Scene object data or null if not found
 */
export function getSceneObject(id) {
  return sceneObjects[id] || null;
}

/**
 * Get all scene object IDs
 * @returns {Array<string>} Array of all object IDs
 */
export function getAllSceneObjectIds() {
  return Object.keys(sceneObjects);
}

/**
 * Get all objects of a specific type
 * @param {string} type - Object type ('splat', 'gltf', etc.)
 * @returns {Array<Object>} Array of scene objects matching the type
 */
export function getSceneObjectsByType(type) {
  return Object.values(sceneObjects).filter((obj) => obj.type === type);
}

/**
 * Get all objects that have envMapWorldCenter defined
 * Useful for finding capture positions for environment maps
 * @returns {Array<Object>} Array of { id, position } objects
 */
export function getEnvMapWorldCenters() {
  const centers = [];
  for (const [id, obj] of Object.entries(sceneObjects)) {
    if (obj.envMapWorldCenter) {
      centers.push({
        id: id,
        position: obj.envMapWorldCenter,
      });
    }
  }
  return centers;
}

/**
 * Console helper: Capture environment map at a named scene's envMapWorldCenter
 * @param {string} sceneId - ID of scene object with envMapWorldCenter (e.g., "interior", "officeHell")
 * @param {Object} options - Additional options (hideObjectIds, filename)
 * @returns {Promise<THREE.Texture>} The captured environment map
 */
export async function captureEnvMapAtScene(sceneId, options = {}) {
  if (!window.sceneManager) {
    throw new Error("window.sceneManager not available");
  }

  const sceneObj = sceneObjects[sceneId];
  if (!sceneObj) {
    throw new Error(`Scene "${sceneId}" not found`);
  }

  if (!sceneObj.envMapWorldCenter) {
    throw new Error(
      `Scene "${sceneId}" does not have envMapWorldCenter defined`
    );
  }

  return window.sceneManager.captureEnvMap({
    position: sceneObj.envMapWorldCenter,
    ...options,
  });
}

export default sceneObjects;
