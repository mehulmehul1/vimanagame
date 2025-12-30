/**
 * shaderHelper.js - THREE.JS SHADER INJECTION UTILITIES
 * =============================================================================
 *
 * ROLE: Utilities for injecting custom shader code into Three.js materials
 * via the onBeforeCompile hook. Used for custom visual effects.
 *
 * KEY FUNCTIONS:
 * - setupUniforms: Add uniforms to a shader
 * - setupShaderSnippets: Inject vertex/fragment code snippets
 *
 * INJECTION POINTS:
 * - vertexGlobal: Outside main() in vertex shader
 * - vertexMain: Inside main() in vertex shader
 * - fragmentGlobal: Outside main() in fragment shader
 * - fragmentMain: Inside main() in fragment shader (after #include <dithering_fragment>)
 *
 * =============================================================================
 */

/**
 * Setup uniforms for a shader
 * @param {Object} shader - Three.js shader object from onBeforeCompile
 * @param {Object} uniforms - Object with uniform definitions
 */
export function setupUniforms(shader, uniforms) {
  const keys = Object.keys(uniforms);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    shader.uniforms[key] = uniforms[key];
  }
}

/**
 * Setup shader code snippets by injecting into Three.js material shader
 * @param {Object} shader - Three.js shader object from onBeforeCompile
 * @param {string} vertexGlobal - Code to inject outside main() in vertex shader
 * @param {string} vertexMain - Code to inject inside main() in vertex shader
 * @param {string} fragmentGlobal - Code to inject outside main() in fragment shader
 * @param {string} fragmentMain - Code to inject inside main() in fragment shader
 */
export function setupShaderSnippets(
  shader,
  vertexGlobal,
  vertexMain,
  fragmentGlobal,
  fragmentMain
) {
  // Vertex shader outside main
  shader.vertexShader = shader.vertexShader.replace(
    "#include <common>",
    `#include <common>
            ${vertexGlobal}
        `
  );

  // Vertex shader inside main
  shader.vertexShader = shader.vertexShader.replace(
    "#include <begin_vertex>",
    `#include <begin_vertex>
            ${vertexMain}
        `
  );

  // Fragment shader outside main
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <common>",
    `#include <common>
            ${fragmentGlobal}
        `
  );

  // Fragment shader inside main
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <dithering_fragment>",
    `#include <dithering_fragment>
            ${fragmentMain}
        `
  );
}

export default {
  setupUniforms,
  setupShaderSnippets,
};
