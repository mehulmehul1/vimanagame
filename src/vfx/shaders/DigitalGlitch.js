/**
 * Digital Glitch Shader
 * Creates a digital glitch effect with displacement and distortion
 */

export const DigitalGlitch = {
  uniforms: {
    tDiffuse: { value: null },
    tDisp: { value: null },
    seed: { value: 0.0 },
    byp: { value: 0 },
    amount: { value: 0.08 },
    angle: { value: 0.02 },
    seed_x: { value: 0.02 },
    seed_y: { value: 0.02 },
    distortion_x: { value: 0.5 },
    distortion_y: { value: 0.6 },
  },

  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform sampler2D tDisp;
    uniform float seed;
    uniform int byp;
    uniform float amount;
    uniform float angle;
    uniform float seed_x;
    uniform float seed_y;
    uniform float distortion_x;
    uniform float distortion_y;
    
    varying vec2 vUv;
    
    float rand(vec2 co) {
      return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }
    
    void main() {
      if (byp == 1) {
        gl_FragColor = texture2D(tDiffuse, vUv);
      } else {
        vec2 p = vUv;
        float xs = floor(gl_FragCoord.x / 0.5);
        float ys = floor(gl_FragCoord.y / 0.5);
        
        vec4 normal = texture2D(tDisp, p * seed * seed);
        vec4 normal2 = texture2D(tDisp, p * seed * seed + 1.0);
        
        p.x += normal.x * distortion_x;
        p.y += normal.y * distortion_y;
        
        vec2 offset = amount * vec2(cos(angle), sin(angle));
        p += offset;
        
        gl_FragColor = texture2D(tDiffuse, p);
        
        vec2 block = vec2(floor(p.x * 10.0) / 10.0, floor(p.y * 10.0) / 10.0);
        gl_FragColor.r += rand(block) * seed_x;
        gl_FragColor.g += rand(block + 1.0) * seed_x;
        gl_FragColor.b += rand(block + 2.0) * seed_x;
        
        gl_FragColor.rgb += mix(-amount, amount, rand(block + seed)) * seed_y;
      }
    }
  `,
};

export default DigitalGlitch;
