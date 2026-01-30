/**
 * StringHighlight TSL Shader - Glow effect on harp strings
 *
 * TSL (WebGPU) implementation with automatic GLSL fallback
 */

import * as THREE from 'three';
import {
    float, vec3, vec4,
    uniform, uv, positionLocal, normalLocal,
    sin, pow, max, dot, normalize,
    Fn
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

// Uniforms
const uTime = uniform(0);
const uIntensity = uniform(0);
const uColor = uniform(new THREE.Color(0x00ffff));
const uCameraPosition = uniform(new THREE.Vector3());

/**
 * Fresnel calculation
 */
const fresnel = Fn(([viewDir, normal]) => {
    return pow(float(1.0).sub(max(dot(viewDir, normal), 0)), 3.0);
});

/**
 * String glow fragment shader
 */
const stringGlowFragment = Fn(() => {
    // Animated pulse traveling up the string
    const pulse = sin(uv().y.mul(10.0).sub(uTime.mul(5.0))).mul(0.5).add(0.5);
    const pulseSquared = pulse.mul(pulse);

    // Fresnel-based rim lighting (would be computed in vertex for accuracy)
    const baseFresnel = float(0.5); // Simplified for TSL

    // Combine effects
    const glow = baseFresnel.mul(0.7).add(pulseSquared.mul(0.3));

    // Apply intensity
    const color = vec3(uColor).mul(glow).mul(uIntensity);

    // Alpha based on intensity and fresnel
    const alpha = baseFresnel.mul(0.5).add(0.3).mul(uIntensity);

    return vec4(color, alpha);
});

/**
 * StringHighlightMaterialTSL - WebGPU TSL material
 */
export class StringHighlightMaterialTSL extends MeshBasicNodeMaterial {
    public uniforms: {
        uTime: { value: number | any };
        uIntensity: { value: number | any };
        uColor: { value: THREE.Color | any };
        uCameraPosition: { value: THREE.Vector3 | any };
    };

    constructor(params: any = {}) {
        super(params);

        this.colorNode = stringGlowFragment();
        this.transparent = true;
        this.depthWrite = false;
        this.blending = THREE.AdditiveBlending;
        this.side = THREE.DoubleSide;

        this.uniforms = {
            uTime,
            uIntensity,
            uColor,
            uCameraPosition
        };
    }

    setTime(value: number) {
        uTime.value = value;
    }

    setIntensity(value: number) {
        uIntensity.value = value;
    }

    setColor(color: THREE.Color) {
        uColor.value.copy(color);
    }

    setCameraPosition(position: THREE.Vector3) {
        uCameraPosition.value.copy(position);
    }

    getIntensity(): number {
        return uIntensity.value;
    }
}

export { uTime, uIntensity, uColor, uCameraPosition };
export default StringHighlightMaterialTSL;
