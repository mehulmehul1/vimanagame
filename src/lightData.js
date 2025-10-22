import { sceneObjects } from "./sceneData.js";
import { GAME_STATES } from "./gameData.js";

/**
 * Light Data - Defines all lights in the scene
 *
 * Supports both regular Three.js lights and Spark splat lights
 *
 * Each light can have criteria to control when it's active based on game state.
 * Criteria uses the same format as sceneData.js (see criteriaHelper.js)
 */

export const lights = {
  // Standard Three.js Lights
  ambient: {
    id: "ambient",
    type: "AmbientLight",
    color: 0xffffff,
    intensity: 0.5,
  },

  mainDirectional: {
    id: "main-directional",
    type: "DirectionalLight",
    color: 0xffffff,
    intensity: 0.8,
    position: { x: 10, y: 20, z: 10 },
    castShadow: true,
  },

  // Splat-based Lights (using SplatEditSdf)
  streetLight: {
    id: "street-light",
    type: "SplatLight",
    splatType: "SPHERE",
    color: { r: 0.9, g: 0.9, b: 0.9 },
    position: { x: -0.84, y: 0.99, z: 64.97 },
    rotation: { x: -Math.PI / 2, y: 0, z: 0 }, // Point downward (-90° rotation from +Z to -Y)
    radius: 3, // Half-angle = π/4 × 0.8 ≈ 36° (72° total cone - typical streetlight)
    opacity: 0.05, // With ADD_RGBA, 0 opacity gives best falloff
    rgbaBlendMode: "ADD_RGBA",
    sdfSmooth: 0.1,
    softEdge: 3, // Larger soft edge for gradual streetlight falloff
    threeLightDuplicate: true, // Creates a Three.js PointLight at the same position
  },

  streetLight2: {
    id: "street-light-2",
    type: "SplatLight",
    splatType: "SPHERE",
    color: { r: 0.9, g: 0.9, b: 0.9 },
    position: { x: 13.03, y: 1.44, z: 75.07 },
    rotation: { x: -1.57, y: 0, z: 0 },
    radius: 5, // Sphere radius
    opacity: 0.025, // With ADD_RGBA, 0 opacity gives best falloff
    rgbaBlendMode: "ADD_RGBA",
    sdfSmooth: 0.1,
    softEdge: 3.0, // Larger soft edge for gradual streetlight falloff
  },

  // Car headlights (parented to GLTF node "Old_Car_01" inside scene object id "car")
  // Scene splats have editable:false, so this only affects fog
  carHeadlight: {
    id: "car-headlight",
    type: "SplatLight",
    splatType: "INFINITE_CONE",
    color: { r: 0.9, g: 0.9, b: 0.9 },
    attachTo: { objectId: "car", childName: "Old_Car_01" },
    position: { x: 0, y: 1, z: 1 },
    rotation: { x: 0.12, y: Math.PI + 0.06, z: -2 },
    radius: 0.2,
    opacity: 0.4, // Can be high since scene splats are protected
    rgbaBlendMode: "ADD_RGBA",
    softEdge: 0.75,
    threeLightDuplicate: {
      type: "PointLight",
      intensity: 100,
      distance: 20,
      castShadow: false,
      position: { x: 0, y: 2, z: 3 },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.DRIVE_BY_PREAMBLE,
        $lt: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  // Office Interior Lights
  officeHemisphere: {
    id: "office-hemisphere",
    type: "HemisphereLight",
    skyColor: 0xe6d4a0, // Warm golden ceiling
    groundColor: 0x3a5a54, // Deep teal floor
    intensity: 0.8,
    position: { x: -4.5, y: 3, z: 85 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  officeDirectional: {
    id: "office-directional",
    type: "DirectionalLight",
    color: 0xffffff,
    intensity: 0.5,
    position: { x: -2, y: 5, z: 88 },
    castShadow: true,
    shadow: {
      mapSize: { width: 1024, height: 1024 },
      camera: {
        left: -5,
        right: 5,
        top: 5,
        bottom: -5,
        near: 0.5,
        far: 15,
      },
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  officeLampKey: {
    id: "office-lamp-key",
    type: "PointLight",
    color: 0xffcc66, // Warm amber
    intensity: 80,
    distance: 12,
    decay: 2,
    position: { x: -6, y: 1.8, z: 86 },
    castShadow: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  officeLampFill: {
    id: "office-lamp-fill",
    type: "PointLight",
    color: 0xffd98c,
    intensity: 60,
    distance: 12,
    decay: 2,
    position: { x: -3, y: 1.8, z: 86 },
    castShadow: false,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },
};

export default lights;
