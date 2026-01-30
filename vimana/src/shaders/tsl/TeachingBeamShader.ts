/**
 * TeachingBeam TSL Shader - Energy beam connecting jelly to string
 *
 * TSL (WebGPU) implementation with automatic GLSL fallback
 */

import * as THREE from 'three';
import {
    float, vec3, vec4,
    uniform, uv,
    sin, pow, abs, mod, smoothstep, Fn
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

// Uniforms
const uTime = uniform(0);
const uIntensity = uniform(0);
const uColor = uniform(new THREE.Color(0x00ffff));
const uCameraPosition = uniform(new THREE.Vector3());

/**
 * Teaching beam fragment shader
 */
const teachingBeamFragment = Fn(() => {
    const vUv = uv();

    // Scrolling effect for energy flow
    const flow = mod(vUv.y.mul(4.0).sub(uTime.mul(3.0)), 1.0);
    const pulse = sin(flow.mul(6.28)).mul(0.5).add(0.5);

    // Core beam
    const core = float(1.0).sub(abs(vUv.x.sub(0.5)).mul(2.0));
    const coreCubed = pow(core, 3.0);

    // Combine effects
    const beam = coreCubed.mul(0.6).add(float(0.2)); // Simplified fresnel
    const beamFinal = beam.add(pulse.mul(0.3));

    // Color with intensity
    const color = vec3(uColor).mul(beamFinal).mul(uIntensity);

    // Alpha fades at ends
    const endFade = smoothstep(0.0, 0.1, vUv.y).mul(smoothstep(1.0, 0.9, vUv.y));
    const alpha = beamFinal.mul(endFade).mul(uIntensity);

    return vec4(color, alpha);
});

/**
 * TeachingBeamMaterialTSL - WebGPU TSL material
 */
export class TeachingBeamMaterialTSL extends MeshBasicNodeMaterial {
    public uniforms: {
        uTime: { value: number | any };
        uIntensity: { value: number | any };
        uColor: { value: THREE.Color | any };
        uCameraPosition: { value: THREE.Vector3 | any };
    };

    constructor(params: any = {}) {
        super(params);

        this.colorNode = teachingBeamFragment();
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;
        this.blending = THREE.AdditiveBlending;

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
export default TeachingBeamMaterialTSL;
