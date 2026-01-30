import * as THREE from 'three';
import {
    float, vec3, vec4,
    uniform, uv, positionLocal,
    sin, atan, log, length, mix, smoothstep,
    Fn // Fn is just a function wrapper in recent TSL
} from 'three/tsl';
import { SpriteNodeMaterial } from 'three/webgpu';

// Uniforms matching WhiteFlashEnding.ts
const uProgress = uniform(0);
const uIntensity = uniform(0);
const uColor = uniform(new THREE.Color(1, 1, 1));

// Spiral SDF function
// float spiralSDF(vec2 uv, float turns, float scale)
const spiralSDF = Fn(([uvInput_imm, turns_imm, scale_imm]) => {
    // TSL functions expect Node representations, so we treat arguments as nodes
    const uvInput = uvInput_imm;
    const turns = turns_imm;
    const scale = scale_imm;

    const centered = uvInput.sub(0.5);
    const angle = atan(centered.y, centered.x);
    const radius = length(centered).mul(2.0);
    // sin(angle * turns + log(radius + 0.1) * scale)
    const logRadius = log(radius.add(0.1));
    return sin(angle.mul(turns).add(logRadius.mul(scale)));
});

// Main Fragment Logic
const whiteFlashFragment = Fn(() => {
    const vUv = uv();

    // Create multiple spirals
    // float spiral1 = spiralSDF(uv, 6.0, 3.0 + uProgress * 2.0);
    const scale1 = float(3.0).add(uProgress.mul(2.0));
    const spiral1 = spiralSDF(vUv, float(6.0), scale1);

    // float spiral2 = spiralSDF(uv, 8.0, -4.0 - uProgress * 3.0);
    const scale2 = float(-4.0).sub(uProgress.mul(3.0));
    const spiral2 = spiralSDF(vUv, float(8.0), scale2);

    // float spiral3 = spiralSDF(uv, 10.0, 5.0 + uProgress * 4.0);
    const scale3 = float(5.0).add(uProgress.mul(4.0));
    const spiral3 = spiralSDF(vUv, float(10.0), scale3);

    // Combine spirals
    // float combined = (spiral1 + spiral2 * 0.5 + spiral3 * 0.25) / 1.75;
    const combined = spiral1.add(spiral2.mul(0.5)).add(spiral3.mul(0.25)).div(1.75);

    // Normalize to 0-1
    const spiralValue = combined.mul(0.5).add(0.5);

    // Distance from center for vignette/intensity falloff
    const dist = length(vUv.sub(0.5)).mul(2.0);

    // float intensity = (1.0 - dist) * spiralValue;
    // Use toVar() if reassigning, but here we can just chain a new variable
    const intensityRaw = float(1.0).sub(dist).mul(spiralValue);

    // intensity = smoothstep(0.0, 1.0, intensity);
    const intensitySmooth = smoothstep(0.0, 1.0, intensityRaw);

    // Fade intensity with progress
    // intensity *= (1.0 - uProgress * 0.5);
    const intensityFinal = intensitySmooth.mul(float(1.0).sub(uProgress.mul(0.5)));

    // Base Color calculation
    const baseColor = vec3(uColor);

    // Add the spiral intensity
    // color += vec3(intensity);
    const colorWithSpiral = baseColor.add(vec3(intensityFinal));

    // Apply White Wash effect
    // float whiteWash = smoothstep(0.6, 1.0, uProgress);
    const whiteWash = smoothstep(0.6, 1.0, uProgress);
    // color = mix(color, vec3(1.0), whiteWash * whiteWash);
    const colorWashed = mix(colorWithSpiral, vec3(1.0), whiteWash.mul(whiteWash));

    // Apply Vignette
    // float vignette = 1.0 - dist * 0.5 * (0.3 + uProgress * 0.5);
    const vignetteFactor = float(0.3).add(uProgress.mul(0.5));
    const vignette = float(1.0).sub(dist.mul(0.5).mul(vignetteFactor));
    const colorVignette = colorWashed.mul(vignette);

    // Apply overall intensity (exposure/brightness)
    const finalColor = colorVignette.mul(uIntensity);

    // Alpha calculation
    // if (uProgress > 0.95) alpha = 1.0 - smoothstep(0.95, 1.0, uProgress);
    const fadeOut = smoothstep(0.95, 1.0, uProgress);
    const alpha = float(1.0).sub(fadeOut);

    return vec4(finalColor, alpha);
});

export class WhiteFlashMaterialTSL extends SpriteNodeMaterial {
    // Define uniforms explicitly for access
    public uniforms: {
        uProgress: { value: number | any }, // TSL UniformNode
        uIntensity: { value: number | any },
        uColor: { value: THREE.Color | any }
    };

    constructor(params: any = {}) {
        super(params);

        this.colorNode = whiteFlashFragment();
        this.positionNode = positionLocal;

        // Expose uniforms for easy access
        // Note: In TSL materials, uniforms are nodes. We store references to them.
        this.uniforms = {
            uProgress,
            uIntensity,
            uColor
        };

        this.transparent = true;
        this.depthWrite = false;
        this.depthTest = false;
    }

    setProgress(value: number) {
        uProgress.value = value;
    }

    setIntensity(value: number) {
        uIntensity.value = value;
    }

    setColor(color: THREE.Color) {
        uColor.value.copy(color);
    }
}

export { uProgress, uIntensity, uColor };
