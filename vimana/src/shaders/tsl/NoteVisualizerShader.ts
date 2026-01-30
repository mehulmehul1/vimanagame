/**
 * NoteVisualizer TSL Shader - Vertical bars for note visualization
 *
 * TSL (WebGPU) implementation with automatic GLSL fallback
 */

import * as THREE from 'three';
import {
    float, vec3, vec4,
    uniform, uv,
    pow, add, Fn
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

// Uniforms
const uTime = uniform(0);
const uColor = uniform(new THREE.Color(0x00ffff));
const uIntensity = uniform(0);
const uCameraPosition = uniform(new THREE.Vector3());

/**
 * Note bar fragment shader
 */
const noteBarFragment = Fn(() => {
    // Fresnel edge glow (simplified)
    const fresnel = float(0.4);

    // Vertical gradient (brighter at top)
    const verticalGrad = uv().y;

    // Combine
    let color = vec3(uColor);
    color = color.add(vec3(1.0).mul(fresnel.mul(0.5)));
    color = color.mul(uIntensity);

    const alpha = float(0.3).add(verticalGrad.mul(0.5)).add(fresnel.mul(0.2)).mul(uIntensity);

    return vec4(color, alpha);
});

/**
 * NoteVisualizerMaterialTSL - WebGPU TSL material
 */
export class NoteVisualizerMaterialTSL extends MeshBasicNodeMaterial {
    public uniforms: {
        uTime: { value: number | any };
        uColor: { value: THREE.Color | any };
        uIntensity: { value: number | any };
        uCameraPosition: { value: THREE.Vector3 | any };
    };

    constructor(params: any = {}) {
        super(params);

        this.colorNode = noteBarFragment();
        this.transparent = true;
        this.depthWrite = false;
        this.blending = THREE.AdditiveBlending;

        this.uniforms = {
            uTime,
            uColor,
            uIntensity,
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

export { uTime, uColor, uIntensity, uCameraPosition };
export default NoteVisualizerMaterialTSL;
