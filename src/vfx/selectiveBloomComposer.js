/**
 * selectiveBloomComposer.js - SELECTIVE BLOOM POST-PROCESSING
 * =============================================================================
 *
 * ROLE: Two-pass rendering for selective bloom effect. Only specified objects
 * receive bloom, composited over the base scene.
 *
 * KEY RESPONSIBILITIES:
 * - Create bloom render pass with UnrealBloomPass
 * - Composite bloom texture over base scene
 * - Configurable bloom strength, threshold, radius
 * - Resolution scaling for performance
 *
 * USAGE:
 *   const bloomComposer = createSelectiveBloomComposer(renderer, scene, camera);
 *   bloomComposer.composer1.render(); // Bloom pass
 *   bloomComposer.composer2.render(); // Composite pass
 *
 * =============================================================================
 */

import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";

/**
 * Create selective bloom composer setup
 * @param {THREE.WebGLRenderer} renderer - WebGL renderer
 * @param {THREE.Scene} scene - Scene to render
 * @param {THREE.Camera} camera - Camera to use
 * @param {Object} options - Configuration options
 * @returns {Object} Object with composer1, composer2, shaderPass, and helper functions
 */
export function createSelectiveBloomComposer(
  renderer,
  scene,
  camera,
  options = {}
) {
  const {
    bloomStrength = 12.0,
    bloomThreshold = 0.5,
    bloomRadius = 0.4,
    bloomResolution = 0.2,
  } = options;

  // Get canvas size
  const canvas = renderer.domElement;
  const res = new THREE.Vector2(canvas.clientWidth, canvas.clientHeight);

  // Create two composers
  const effectComposer1 = new EffectComposer(renderer);
  const effectComposer2 = new EffectComposer(renderer);

  // Render pass
  const renderPass = new RenderPass(scene, camera);

  // Bloom pass (applied to first composer)
  const bloomPass = new UnrealBloomPass(
    res,
    bloomStrength,
    bloomRadius,
    bloomThreshold
  );

  // Output pass
  const outPass = new OutputPass();

  // Setup first composer (bloom only)
  effectComposer1.addPass(renderPass);
  effectComposer1.addPass(bloomPass);
  effectComposer1.renderToScreen = false;

  // Create shader pass to composite bloom with base scene
  const shaderPass = new ShaderPass(
    new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null }, // EffectComposer will set this
        uBloomTexture: {
          value: effectComposer1.renderTarget2.texture,
        },
        uStrength: {
          value: bloomStrength,
        },
      },

      vertexShader: `
        varying vec2 vUv;
        void main(){
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
        }
    `,

      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D uBloomTexture;
        uniform float uStrength;
        varying vec2 vUv;
        void main(){
            vec4 baseEffect = texture2D(tDiffuse,vUv);
            vec4 bloomEffect = texture2D(uBloomTexture,vUv);
            gl_FragColor = baseEffect + bloomEffect * uStrength;
        }
    `,
    })
  );

  // Setup second composer (composite)
  effectComposer2.addPass(renderPass);
  effectComposer2.addPass(shaderPass);
  effectComposer2.addPass(outPass);

  /**
   * Update bloom strength
   * @param {number} strength - Bloom strength (0-20)
   */
  function updateBloomStrength(strength) {
    shaderPass.uniforms.uStrength.value = strength;
  }

  /**
   * Update bloom parameters
   * @param {Object} params - Bloom parameters
   */
  function updateBloomParams(params) {
    if (params.strength !== undefined) {
      shaderPass.uniforms.uStrength.value = params.strength;
    }
    if (params.threshold !== undefined) {
      bloomPass.threshold = params.threshold;
    }
    if (params.radius !== undefined) {
      bloomPass.radius = params.radius;
    }
  }

  /**
   * Handle window resize
   * @param {number} width - New width
   * @param {number} height - New height
   */
  function setSize(width, height) {
    renderPass.setSize(width, height);
    bloomPass.setSize(width, height);
    shaderPass.setSize(width, height);
    outPass.setSize(width, height);
    effectComposer1.setSize(width, height);
    effectComposer2.setSize(width, height);
  }

  /**
   * Dispose of resources
   */
  function dispose() {
    effectComposer1.dispose();
    effectComposer2.dispose();
  }

  return {
    composer1: effectComposer1,
    composer2: effectComposer2,
    shaderPass,
    updateBloomStrength,
    updateBloomParams,
    setSize,
    dispose,
  };
}

export default createSelectiveBloomComposer;
