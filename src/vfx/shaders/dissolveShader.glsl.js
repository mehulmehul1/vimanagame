/**
 * Dissolve Shader Snippets
 *
 * Edge-based dissolve effect using Perlin noise.
 * Injects into Three.js materials via onBeforeCompile.
 */

export const vertexGlobal = `
    varying vec3 vPos;
    varying vec3 vWorldPos;
`;

export const vertexMain = `
    vPos = position;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
`;

export const fragmentGlobal = `
    varying vec3 vPos;
    varying vec3 vWorldPos;
    uniform vec3 uEdgeColor1;
    uniform vec3 uEdgeColor2;
    uniform float uFreq;
    uniform float uAmp;
    uniform float uProgress;
    uniform float uEdge;
    uniform float uDissolveMode; // 0 = noise, 1 = wipe
    uniform float uWipeDirection; // 0 = bottom-to-top, 1 = top-to-bottom
    uniform float uWipeSoftness;
    uniform vec2 uWipeBounds; // x = min Y, y = max Y (world space)
`;

export const fragmentMain = `
    float dissolveValue;
    
    if (uDissolveMode < 0.5) {
        // Noise-based dissolve
        dissolveValue = cnoise(vPos * uFreq) * uAmp;
    } else {
        // Wipe-based dissolve (bottom-to-top or top-to-bottom)
        // Normalize world Y position to 0-1 range based on object bounds
        float normalizedY = (vWorldPos.y - uWipeBounds.x) / (uWipeBounds.y - uWipeBounds.x);
        normalizedY = clamp(normalizedY, 0.0, 1.0);
        
        // For bottom-to-top: bottom should have HIGH dissolveValue (appears first when progress decreases)
        // For top-to-bottom: top should have HIGH dissolveValue
        if (uWipeDirection < 0.5) {
            // Bottom-to-top: invert so bottom = 1, top = 0
            normalizedY = 1.0 - normalizedY;
        }
        
        // Map to -14 to +14 range (high normalizedY = high dissolveValue)
        dissolveValue = normalizedY * 28.0 - 14.0;
        
        // Add softness (subtle noise)
        float softEdge = uWipeSoftness * 2.0;
        dissolveValue += (fract(sin(vPos.x * 12.9898 + vPos.z * 78.233) * 43758.5453) - 0.5) * softEdge;
    }

    if(dissolveValue < uProgress) discard;

    float edgeWidth = uProgress + uEdge;
    if(dissolveValue > uProgress && dissolveValue < edgeWidth){
        gl_FragColor = vec4(uEdgeColor1, dissolveValue);
    }
    gl_FragColor = vec4(gl_FragColor.xyz, 1.0);
`;

export default {
  vertexGlobal,
  vertexMain,
  fragmentGlobal,
  fragmentMain,
};
