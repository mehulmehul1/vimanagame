import * as THREE from "three";

/**
 * Create a custom shader material for headlight beams
 * Fades transparency based on local Z position (further forward = more transparent)
 * @param {THREE.Material} originalMaterial - Original material from GLTF
 * @returns {THREE.ShaderMaterial}
 */
export function createHeadlightBeamShader(originalMaterial) {
  // Extract properties from original material
  const color = originalMaterial.color || new THREE.Color(0xffffff);
  const emissive = originalMaterial.emissive || new THREE.Color(0xffffff);
  const emissiveIntensity = originalMaterial.emissiveIntensity || 1.0;
  const opacity =
    originalMaterial.opacity !== undefined ? originalMaterial.opacity : 1.0;

  const shaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
      baseColor: { value: color.clone() },
      emissiveColor: { value: emissive.clone() },
      emissiveIntensity: { value: emissiveIntensity },
      baseOpacity: { value: opacity * 0.1 }, // Lower alpha at solid end
      fadeStart: { value: 0.0 }, // Z position where fade starts
      fadeEnd: { value: 10.0 }, // Z position where completely transparent
    },
    vertexShader: `
      varying vec3 vPosition;
      
      void main() {
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 baseColor;
      uniform vec3 emissiveColor;
      uniform float emissiveIntensity;
      uniform float baseOpacity;
      uniform float fadeStart;
      uniform float fadeEnd;
      
      varying vec3 vPosition;
      
      void main() {
        // Calculate fade based on local Z position (further = more transparent)
        float zFade = smoothstep(fadeStart, fadeEnd, vPosition.z);
        
        // Combine base color with emissive
        vec3 finalColor = baseColor + (emissiveColor * emissiveIntensity);
        
        // Apply fade to opacity
        float finalOpacity = baseOpacity * (1.0 - zFade);
        
        gl_FragColor = vec4(finalColor, finalOpacity);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    side: originalMaterial.side || THREE.FrontSide,
    // Polygon offset to help with z-fighting and depth sorting with splats
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });

  console.log(
    `HeadlightBeamShader: Created shader (fade: ${shaderMaterial.uniforms.fadeStart.value} to ${shaderMaterial.uniforms.fadeEnd.value})`
  );

  return shaderMaterial;
}
