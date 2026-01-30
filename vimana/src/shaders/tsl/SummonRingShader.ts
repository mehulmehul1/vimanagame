/**
 * SummonRing TSL Shader - Expanding ring effect before jelly emerges
 *
 * TSL (WebGPU) implementation with automatic GLSL fallback
 */

import * as THREE from 'three';
import {
    float, vec2, vec3, vec4,
    uniform, uv,
    distance, smoothstep,
    Fn
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

// Uniforms
const uTime = uniform(0);
const uDuration = uniform(0.6);
const uColor = uniform(new THREE.Color(0x00ffff));

/**
 * Ring fragment shader in TSL
 */
const summonRingFragment = Fn(() => {
    const vUv = uv();

    // Distance from center (0.5, 0.5 is center)
    const center = vec2(0.5, 0.5);
    const dist = distance(vUv, center);

    // Progress (0 to 1)
    const progress = uTime.div(uDuration);

    // Create expanding ring
    const ringWidth = float(0.15);
    const ringCenter = progress.mul(0.8);
    const ring = smoothstep(ringCenter.add(ringWidth), ringCenter, dist).mul(
        smoothstep(ringCenter.sub(ringWidth), ringCenter, dist)
    );

    // Fade out at edges
    const alpha = ring.mul(float(1.0).sub(progress));

    // Inner glow
    const innerGlow = smoothstep(0.3, 0.0, dist).mul(progress.mul(0.5));

    const finalColor = vec3(uColor).mul(ring.add(innerGlow));
    const finalAlpha = alpha.mul(0.8).add(innerGlow.mul(0.3));

    return vec4(finalColor, finalAlpha);
});

/**
 * SummonRingMaterialTSL - WebGPU TSL material
 */
export class SummonRingMaterialTSL extends MeshBasicNodeMaterial {
    public uniforms: {
        uTime: { value: number | any };
        uDuration: { value: number | any };
        uColor: { value: THREE.Color | any };
    };

    constructor(params: any = {}) {
        super(params);

        this.colorNode = summonRingFragment();
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;
        this.blending = THREE.AdditiveBlending;

        this.uniforms = {
            uTime,
            uDuration,
            uColor
        };
    }

    setTime(value: number) {
        uTime.value = value;
    }

    setDuration(value: number) {
        uDuration.value = value;
    }

    setColor(color: THREE.Color) {
        uColor.value.copy(color);
    }

    getTime(): number {
        return uTime.value;
    }
}

export { uTime, uDuration, uColor };
export default SummonRingMaterialTSL;
