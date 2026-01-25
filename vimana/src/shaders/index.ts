// ============================================================================
// WATER & VORTEX SHADERS (Story 1.1)
// ============================================================================

export const waterVertexShader = `
uniform float uTime;
uniform float uHarpFrequencies[6];
uniform float uHarpVelocities[6];
uniform float uDuetProgress;
uniform float uShipPatience;
uniform vec3 uBioluminescentColor;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vViewPosition;
varying float vWaveHeight;
varying vec3 vBioluminescence;

float simplexNoise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float perlin(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = simplexNoise(i);
    float b = simplexNoise(i + vec2(1.0, 0.0));
    float c = simplexNoise(i + vec2(0.0, 1.0));
    float d = simplexNoise(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
        if (i >= octaves) break;
        value += amplitude * perlin(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec2 waveCoord = position.xz * 0.5;
    float baseWave = fbm(waveCoord + uTime * 0.1, 3) * 0.15;
    float stringResonance = 0.0;
    vec3 totalBioluminescence = vec3(0.0);
    for (int i = 0; i < 6; i++) {
        vec2 stringOrigin = vec2(float(i) * 2.0 - 5.0, 0.0);
        float distTo = distance(position.xz, stringOrigin);
        float stringWave = sin(distTo * 3.0 - uTime * 2.0 + uHarpFrequencies[i]) * 0.05;
        stringWave *= smoothstep(3.0, 0.0, distTo);
        float intensity = uHarpVelocities[i] * 0.5;
        stringResonance += stringWave * intensity;
        totalBioluminescence += uBioluminescentColor * intensity * (1.0 - distTo / 4.0);
    }
    vWaveHeight = baseWave + stringResonance;
    vWaveHeight *= (1.0 + uDuetProgress * 0.5);
    vec3 newPosition = position;
    newPosition.y += vWaveHeight;
    vBioluminescence = totalBioluminescence * (0.5 + uDuetProgress * 0.5);
    vec4 worldPos = modelMatrix * vec4(newPosition, 1.0);
    vWorldPosition = worldPos.xyz;
    vViewPosition = (viewMatrix * worldPos).xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const waterFragmentShader = `
uniform float uTime;
uniform float uDuetProgress;
uniform float uShipPatience;
uniform float uHarmonicResonance;
uniform vec3 uBioluminescentColor;
uniform vec3 uCameraPosition;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vViewPosition;
varying float vWaveHeight;
varying vec3 vBioluminescence;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float caustics(vec2 uv, float time) {
    float c = 0.0;
    float scale = 8.0;
    float speed = time * 0.5;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 p = uv * scale * (1.0 + fi * 0.5);
        p += vec2(snoise(p + speed * (0.3 + fi * 0.1)), snoise(p + speed * (0.2 + fi * 0.15) + 100.0));
        c += abs(snoise(p)) * (1.0 / (1.0 + fi));
    }
    return c * 0.5;
}

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);
    vec3 deepColor = vec3(0.0, 0.1, 0.2);
    vec3 shallowColor = vec3(0.0, 0.3, 0.4);
    vec3 baseColor = mix(deepColor, shallowColor, vWaveHeight * 2.0 + 0.5);
    vec2 causticUv = vWorldPosition.xz * 0.2;
    float caustic = caustics(causticUv, uTime);
    vec3 glow = uBioluminescentColor * (0.3 + caustic * 0.2);
    glow += vBioluminescence * 0.5;
    glow *= (1.0 + uHarmonicResonance * 2.0);
    float duetIntensity = 0.5 + uDuetProgress * 1.5;
    glow *= duetIntensity;
    vec3 finalColor = baseColor + glow;
    vec3 fresnelColor = vec3(0.3, 0.8, 1.0);
    finalColor += fresnelColor * fresnel * 0.3;
    float alpha = 0.6 + fresnel * 0.35;
    alpha *= (0.8 + uHarmonicResonance * 0.2);
    gl_FragColor = vec4(finalColor, alpha);
}
`;

export const vortexVertexShader = `
uniform float uTime;
uniform float uVortexActivation;
uniform float uDuetProgress;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vLocalPosition;
varying float vActivation;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vLocalPosition = position;
    vActivation = uVortexActivation;
    float spinSpeed = 1.0 + uVortexActivation * 3.0;
    float angle = uTime * spinSpeed;
    float breathe = sin(uTime * 2.0) * 0.05 * (0.5 + uVortexActivation * 0.5);
    float dist = length(position.xz);
    float swirlAngle = angle * (1.0 - dist * 0.3);
    mat2 rotation = mat2(cos(swirlAngle), -sin(swirlAngle), sin(swirlAngle), cos(swirlAngle));
    vec3 newPosition = position;
    newPosition.xz = rotation * position.xz;
    newPosition += normal * breathe;
    newPosition += normal * uVortexActivation * 0.1;
    vec4 worldPos = modelMatrix * vec4(newPosition, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const vortexFragmentShader = `
uniform float uTime;
uniform float uVortexActivation;
uniform float uDuetProgress;
uniform vec3 uInnerColor;
uniform vec3 uOuterColor;
uniform vec3 uCoreColor;
uniform vec3 uCameraPosition;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vLocalPosition;
varying float vActivation;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m*m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float swirlNoise(vec3 p, float time) {
    float angle = atan(p.z, p.x);
    float radius = length(p.xz);
    float spiral = angle + radius * 3.0 + time;
    vec3 spiralPos = vec3(cos(spiral) * radius, p.y, sin(spiral) * radius);
    return snoise(spiralPos * 2.0) * 0.5 + 0.5;
}

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);
    float swirl = swirlNoise(vLocalPosition, uTime * (1.0 + uVortexActivation * 2.0));
    vec3 baseColor;
    if (uVortexActivation < 0.5) {
        float t = uVortexActivation * 2.0;
        baseColor = mix(uInnerColor, uOuterColor, t);
    } else {
        float t = (uVortexActivation - 0.5) * 2.0;
        baseColor = mix(uOuterColor, uCoreColor, t);
    }
    baseColor += swirl * 0.2;
    float edgeIntensity = 0.5 + uVortexActivation * 1.5;
    vec3 glow = baseColor * fresnel * edgeIntensity;
    vec3 finalColor = baseColor * 0.6 + glow;
    float emissive = 0.2 + uVortexActivation * 2.8;
    finalColor *= emissive;
    if (uVortexActivation > 0.7) {
        float coreIntensity = (uVortexActivation - 0.7) / 0.3;
        finalColor = mix(finalColor, uCoreColor, coreIntensity * 0.5);
    }
    gl_FragColor = vec4(finalColor, 0.9);
}
`;

// ============================================================================
// JELLY CREATURE SHADERS (Story 1.2)
// ============================================================================

export const jellyVertexShader = `
uniform float uTime;
uniform float uPulseRate;
uniform float uIsTeaching;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec2 vUv;
varying float vPulse;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m*m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    float pulse = sin(uTime * uPulseRate) * 0.5 + 0.5;
    vPulse = pulse;
    float noise = snoise(position * 2.0 + uTime * 0.5) * 0.1;
    float teachingEnhancement = 1.0 + uIsTeaching * 0.5;
    vec3 newPosition = position;
    newPosition += normal * (pulse * 0.15 + noise) * teachingEnhancement;
    float bellFactor = smoothstep(-0.5, 0.5, position.y);
    newPosition += normal * bellFactor * pulse * 0.1;
    vec4 worldPos = modelMatrix * vec4(newPosition, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const jellyFragmentShader = `
uniform float uTime;
uniform float uPulseRate;
uniform float uIsTeaching;
uniform float uTeachingIntensity;
uniform vec3 uBioluminescentColor;
uniform vec3 uCameraPosition;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec2 vUv;
varying float vPulse;

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);
    vec3 baseColor = vec3(0.4, 0.8, 0.7);
    float glowIntensity = 0.3 + vPulse * 0.4 + uTeachingIntensity * 0.3;
    vec3 glow = uBioluminescentColor * glowIntensity;
    float internalVisibility = 1.0 - fresnel * 0.5;
    vec3 internalColor = vec3(0.8, 0.9, 0.85) * internalVisibility * 0.3;
    if (uIsTeaching > 0.5) {
        glow += vec3(0.2, 0.1, 0.0) * uTeachingIntensity;
    }
    vec3 finalColor = baseColor * 0.4 + glow + internalColor;
    vec3 rimColor = uBioluminescentColor * 2.0;
    finalColor += rimColor * fresnel * 0.5;
    float alpha = 0.5 + fresnel * 0.3 + vPulse * 0.1;
    gl_FragColor = vec4(finalColor, alpha);
}
`;

// ============================================================================
// SHELL COLLECTIBLE SHADERS (Story 1.6)
// ============================================================================

export const shellVertexShader = `
uniform float uTime;
uniform float uAppearProgress;
uniform float uDissolveAmount;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vLocalPosition;
varying vec2 vUv;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vLocalPosition = position;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const shellFragmentShader = `
uniform float uTime;
uniform float uAppearProgress;
uniform float uDissolveAmount;
uniform vec3 uCameraPosition;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vLocalPosition;
varying vec2 vUv;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m*m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

vec3 iridescence(vec3 viewDir, vec3 normal, float time) {
    vec3 colors[5];
    colors[0] = vec3(1.0, 0.8, 0.6);
    colors[1] = vec3(0.8, 0.6, 1.0);
    colors[2] = vec3(0.6, 0.9, 1.0);
    colors[3] = vec3(1.0, 0.7, 0.5);
    colors[4] = vec3(0.7, 1.0, 0.8);
    float viewAngle = dot(viewDir, normal);
    float t = sin(viewAngle * 5.0 + time) * 0.5 + 0.5;
    int index = int(t * 4.0);
    float localT = fract(t * 4.0);
    vec3 result = mix(colors[index], colors[index + 1], localT);
    return result;
}

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);
    vec3 baseColor = vec3(0.95, 0.9, 0.85);
    vec3 irid = iridescence(viewDir, vNormal, uTime * 0.5);
    float angle = atan(vLocalPosition.z, vLocalPosition.x);
    float radius = length(vLocalPosition.xz);
    float spiral = sin(angle * 3.0 + radius * 5.0) * 0.5 + 0.5;
    vec3 finalColor = mix(baseColor, irid, spiral * 0.6 + fresnel * 0.4);
    finalColor += irid * fresnel * 0.3;
    float dissolve = snoise(vLocalPosition * 3.0 + uTime) * 0.5 + 0.5;
    float alpha = smoothstep(uDissolveAmount - 0.1, uDissolveAmount + 0.1, dissolve);
    alpha *= uAppearProgress;
    gl_FragColor = vec4(finalColor, alpha);
}
`;

// ============================================================================
// WHITE FLASH ENDING SHADERS (Story 1.8)
// ============================================================================

export const whiteFlashVertexShader = `
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const whiteFlashFragmentShader = `
uniform float uTime;
uniform float uProgress;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;

varying vec2 vUv;

float spiralSDF(vec2 uv, float turns, float scale) {
    vec2 centered = uv - 0.5;
    float angle = atan(centered.y, centered.x);
    float radius = length(centered) * 2.0;
    float spiral = sin(angle * turns + log(radius + 0.1) * scale);
    return spiral;
}

void main() {
    vec2 uv = vUv;
    float spiral1 = spiralSDF(uv, 6.0, 3.0 + uProgress * 2.0);
    float spiral2 = spiralSDF(uv, 8.0, -4.0 - uProgress * 3.0);
    float spiral3 = spiralSDF(uv, 10.0, 5.0 + uProgress * 4.0);
    float combined = (spiral1 + spiral2 * 0.5 + spiral3 * 0.25) / 1.75;
    float spiralValue = combined * 0.5 + 0.5;
    float dist = length(uv - 0.5) * 2.0;
    float intensity = (1.0 - dist) * spiralValue;
    intensity = smoothstep(0.0, 1.0, intensity);
    intensity *= (1.0 - uProgress * 0.5);
    vec3 color;
    if (uProgress < 0.5) {
        float t = uProgress * 2.0;
        color = mix(uColor1, uColor2, smoothstep(0.0, 1.0, t));
    } else {
        float t = (uProgress - 0.5) * 2.0;
        color = mix(uColor2, uColor3, smoothstep(0.0, 1.0, t));
    }
    color += vec3(intensity);
    float whiteWash = smoothstep(0.6, 1.0, uProgress);
    color = mix(color, vec3(1.0), whiteWash * whiteWash);
    float vignette = 1.0 - dist * 0.5 * (0.3 + uProgress * 0.5);
    color *= vignette;
    float alpha = 1.0;
    if (uProgress > 0.95) {
        alpha = 1.0 - smoothstep(0.95, 1.0, uProgress);
    }
    gl_FragColor = vec4(color, alpha);
}
`;
