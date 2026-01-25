import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import EditorManager from '../../core/EditorManager';
import './ShaderEditor.css';

/**
 * TSL Shader preset
 */
interface ShaderPreset {
    name: string;
    category: string;
    vertexShader: string;
    fragmentShader: string;
    uniforms?: Record<string, { value: number | number[]; type: 'float' | 'vec2' | 'vec3' | 'vec4' | 'color' }>;
}

/**
 * TSL Shader Presets
 */
const SHADER_PRESETS: ShaderPreset[] = [
    {
        name: 'Basic Lambert',
        category: 'Lighting',
        vertexShader: `varying vec3 vNormal;
varying vec3 vPosition;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
        fragmentShader: `varying vec3 vNormal;
varying vec3 vPosition;

uniform vec3 baseColor;
uniform vec3 lightDirection;
uniform vec3 lightColor;

void main() {
    float NdotL = max(dot(vNormal, lightDirection), 0.0);
    vec3 ambient = baseColor * 0.2;
    vec3 diffuse = baseColor * lightColor * NdotL;
    gl_FragColor = vec4(ambient + diffuse, 1.0);
}`,
        uniforms: {
            baseColor: { value: [0.5, 0.6, 1.0], type: 'color' },
            lightDirection: { value: [0.5, 0.5, 1.0], type: 'vec3' },
            lightColor: { value: [1.0, 1.0, 1.0], type: 'color' }
        }
    },
    {
        name: 'Phong',
        category: 'Lighting',
        vertexShader: `varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec2 vUv;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    vUv = uv;
    gl_Position = projectionMatrix * mvPosition;
}`,
        fragmentShader: `varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec2 vUv;

uniform vec3 baseColor;
uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform float shininess;
uniform float specularStrength;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(lightDirection);
    vec3 viewDir = normalize(vViewPosition);

    // Ambient
    vec3 ambient = baseColor * 0.1;

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * baseColor * lightColor;

    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}`,
        uniforms: {
            baseColor: { value: [0.5, 0.6, 1.0], type: 'color' },
            lightDirection: { value: [0.5, 0.5, 1.0], type: 'vec3' },
            lightColor: { value: [1.0, 1.0, 1.0], type: 'color' },
            shininess: { value: 32.0, type: 'float' },
            specularStrength: { value: 0.5, type: 'float' }
        }
    },
    {
        name: 'Rim Light',
        category: 'Stylized',
        vertexShader: `varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    gl_Position = projectionMatrix * mvPosition;
}`,
        fragmentShader: `varying vec3 vNormal;
varying vec3 vViewPosition;

uniform vec3 baseColor;
uniform vec3 rimColor;
uniform float rimPower;
uniform float rimIntensity;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);

    float rimFactor = 1.0 - max(dot(viewDir, normal), 0.0);
    rimFactor = pow(rimFactor, rimPower);

    vec3 color = baseColor;
    color += rimColor * rimFactor * rimIntensity;

    gl_FragColor = vec4(color, 1.0);
}`,
        uniforms: {
            baseColor: { value: [0.2, 0.4, 0.8], type: 'color' },
            rimColor: { value: [0.3, 0.6, 1.0], type: 'color' },
            rimPower: { value: 3.0, type: 'float' },
            rimIntensity: { value: 0.5, type: 'float' }
        }
    },
    {
        name: 'Toon',
        category: 'Stylized',
        vertexShader: `varying vec3 vNormal;

void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
        fragmentShader: `varying vec3 vNormal;

uniform vec3 lightDirection;
uniform vec3 color1;
uniform vec3 color2;
uniform vec3 color3;
uniform float threshold1;
uniform float threshold2;

void main() {
    float NdotL = dot(vNormal, normalize(lightDirection));

    vec3 color;
    if (NdotL < threshold1) {
        color = color1;
    } else if (NdotL < threshold2) {
        color = color2;
    } else {
        color = color3;
    }

    gl_FragColor = vec4(color, 1.0);
}`,
        uniforms: {
            lightDirection: { value: [0.5, 0.5, 1.0], type: 'vec3' },
            color1: { value: [0.1, 0.1, 0.2], type: 'color' },
            color2: { value: [0.3, 0.3, 0.5], type: 'color' },
            color3: { value: [0.6, 0.6, 0.8], type: 'color' },
            threshold1: { value: 0.3, type: 'float' },
            threshold2: { value: 0.6, type: 'float' }
        }
    },
    {
        name: 'Hologram',
        category: 'Effects',
        vertexShader: `varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;

void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
        fragmentShader: `varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;

uniform float time;
uniform vec3 baseColor;
uniform float scanlineDensity;
uniform float scanlineIntensity;
uniform float flickerSpeed;

void main() {
    // Scanline effect
    float scanline = sin(vUv.y * scanlineDensity + time * 10.0);
    scanline = scanline * 0.5 + 0.5;

    // Hologram flicker
    float flicker = sin(time * flickerSpeed) * 0.5 + 0.5;

    // Rim light
    vec3 viewDir = normalize(cameraPosition - vPosition);
    float rim = 1.0 - max(dot(vNormal, viewDir), 0.0);
    rim = pow(rim, 3.0);

    vec3 color = baseColor;
    color += vec3(scanline * scanlineIntensity);
    color *= 0.8 + flicker * 0.4;
    color += rim * baseColor * 2.0;

    gl_FragColor = vec4(color, 0.8);
}`,
        uniforms: {
            time: { value: 0.0, type: 'float' },
            baseColor: { value: [0.0, 0.8, 1.0], type: 'color' },
            scanlineDensity: { value: 100.0, type: 'float' },
            scanlineIntensity: { value: 0.1, type: 'float' },
            flickerSpeed: { value: 5.0, type: 'float' }
        }
    },
    {
        name: 'Gradient Map',
        category: 'Effects',
        vertexShader: `varying vec2 vUv;
varying vec3 vNormal;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
        fragmentShader: `varying vec2 vUv;
varying vec3 vNormal;

uniform vec3 colorLow;
uniform vec3 colorMid;
uniform vec3 colorHigh;

void main() {
    float NdotL = dot(vNormal, normalize(vec3(1.0, 1.0, 1.0))) * 0.5 + 0.5;

    vec3 color;
    if (NdotL < 0.5) {
        color = mix(colorLow, colorMid, NdotL * 2.0);
    } else {
        color = mix(colorMid, colorHigh, (NdotL - 0.5) * 2.0);
    }

    gl_FragColor = vec4(color, 1.0);
}`,
        uniforms: {
            colorLow: { value: [0.1, 0.1, 0.3], type: 'color' },
            colorMid: { value: [0.4, 0.3, 0.6], type: 'color' },
            colorHigh: { value: [0.9, 0.8, 1.0], type: 'color' }
        }
    }
];

/**
 * Shader Editor Panel - Monaco-based TSL shader editor
 *
 * Features:
 * - Monaco editor for TSL code input
 * - Live preview on sphere in mini-viewport
 * - Export to material format
 * - Preset shaders library
 */
interface ShaderEditorProps {
    editorManager?: EditorManager;
    onExportShader?: (shader: ShaderPreset) => void;
}

const ShaderEditor: React.FC<ShaderEditorProps> = ({ editorManager, onExportShader }) => {
    // Editor state
    const [vertexShader, setVertexShader] = useState(SHADER_PRESETS[0].vertexShader);
    const [fragmentShader, setFragmentShader] = useState(SHADER_PRESETS[0].fragmentShader);
    const [shaderName, setShaderName] = useState(SHADER_PRESETS[0].name);
    const [uniforms, setUniforms] = useState<ShaderPreset['uniforms']>(SHADER_PRESETS[0].uniforms);
    const [selectedPreset, setSelectedPreset] = useState(0);

    // Preview state
    const [time, setTime] = useState(0);
    const previewCanvasRef = useRef<HTMLCanvasElement>(null);
    const previewSceneRef = useRef<{
        scene: THREE.Scene;
        camera: THREE.PerspectiveCamera;
        renderer: THREE.WebGLRenderer;
        sphere: THREE.Mesh;
        material: THREE.ShaderMaterial;
        animationId: number | null;
    } | null>(null);

    // UI state
    const [showPresets, setShowPresets] = useState(true);
    const [activeTab, setActiveTab] = useState<'vertex' | 'fragment'>('fragment');

    /**
     * Initialize preview scene
     */
    useEffect(() => {
        if (!previewCanvasRef.current) return;

        const canvas = previewCanvasRef.current;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        // Camera
        const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
        camera.position.set(0, 0, 2.5);

        // Renderer
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Sphere with shader material
        const geometry = new THREE.SphereGeometry(1, 64, 64);
        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: convertUniforms(uniforms),
            side: THREE.DoubleSide
        });
        const sphere = new THREE.Mesh(geometry, material);
        scene.add(sphere);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        scene.add(ambientLight);

        // Store refs
        previewSceneRef.current = {
            scene,
            camera,
            renderer,
            sphere,
            material,
            animationId: null
        };

        // Animation loop
        const animate = () => {
            setTime((prev) => prev + 0.016);

            // Update time uniform if exists
            if (material.uniforms.time) {
                material.uniforms.time.value = time;
            }

            // Rotate sphere
            sphere.rotation.y += 0.005;

            renderer.render(scene, camera);
            previewSceneRef.current!.animationId = requestAnimationFrame(animate);
        };
        animate();

        // Cleanup
        return () => {
            if (previewSceneRef.current?.animationId) {
                cancelAnimationFrame(previewSceneRef.current.animationId);
            }
            renderer.dispose();
            geometry.dispose();
            material.dispose();
        };
    }, []);

    /**
     * Update shader when code changes
     */
    useEffect(() => {
        if (!previewSceneRef.current) return;

        const { material } = previewSceneRef.current;

        material.vertexShader = vertexShader;
        material.fragmentShader = fragmentShader;
        material.uniforms = convertUniforms(uniforms);
        material.needsUpdate = true;
    }, [vertexShader, fragmentShader, uniforms]);

    /**
     * Convert uniform definitions to THREE uniforms
     */
    const convertUniforms = (uniforms?: ShaderPreset['uniforms']): Record<string, THREE.IUniform> => {
        const converted: Record<string, THREE.IUniform> = {};

        if (uniforms) {
            Object.entries(uniforms).forEach(([name, def]) => {
                let value: any = def.value;

                // Create THREE objects for vectors/colors
                if (def.type === 'vec2' && Array.isArray(value)) {
                    value = new THREE.Vector2(value[0], value[1]);
                } else if (def.type === 'vec3' && Array.isArray(value)) {
                    value = new THREE.Vector3(value[0], value[1], value[2]);
                } else if (def.type === 'vec4' && Array.isArray(value)) {
                    value = new THREE.Vector4(value[0], value[1], value[2], value[3]);
                } else if (def.type === 'color' && Array.isArray(value)) {
                    value = new THREE.Color(value[0], value[1], value[2]);
                }

                converted[name] = { value };
            });
        }

        // Always include time
        if (!converted.time) {
            converted.time = { value: 0 };
        }

        return converted;
    };

    /**
     * Load preset
     */
    const loadPreset = (index: number) => {
        const preset = SHADER_PRESETS[index];
        setShaderName(preset.name);
        setVertexShader(preset.vertexShader);
        setFragmentShader(preset.fragmentShader);
        setUniforms(preset.uniforms);
        setSelectedPreset(index);
    };

    /**
     * Export shader
     */
    const exportShader = () => {
        const shader: ShaderPreset = {
            name: shaderName,
            category: 'Custom',
            vertexShader,
            fragmentShader,
            uniforms
        };

        if (onExportShader) {
            onExportShader(shader);
        } else {
            // Download as JSON
            const blob = new Blob([JSON.stringify(shader, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${shaderName.toLowerCase().replace(/\s+/g, '_')}_shader.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    };

    /**
     * Update uniform value
     */
    const updateUniform = (name: string, value: number | number[]) => {
        if (uniforms) {
            setUniforms({
                ...uniforms,
                [name]: { ...uniforms[name], value }
            });
        }
    };

    /**
     * Group presets by category
     */
    const groupedPresets = SHADER_PRESETS.reduce((acc, preset, index) => {
        if (!acc[preset.category]) {
            acc[preset.category] = [];
        }
        acc[preset.category].push({ ...preset, index });
        return acc;
    }, {} as Record<string, Array<ShaderPreset & { index: number }>>);

    return (
        <div className="shader-editor-container">
            {/* Presets sidebar */}
            {showPresets && (
                <div className="shader-presets">
                    <div className="presets-header">
                        <h4>Presets</h4>
                        <button
                            className="toggle-presets"
                            onClick={() => setShowPresets(false)}
                        >
                            ◀
                        </button>
                    </div>
                    <div className="presets-content">
                        {Object.entries(groupedPresets).map(([category, presets]) => (
                            <div key={category} className="preset-category">
                                <div className="preset-category-header">{category}</div>
                                {presets.map((preset) => (
                                    <button
                                        key={preset.name}
                                        className={`preset-item ${selectedPreset === preset.index ? 'active' : ''}`}
                                        onClick={() => loadPreset(preset.index)}
                                    >
                                        {preset.name}
                                    </button>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Main editor area */}
            <div className="shader-main">
                {/* Header */}
                <div className="shader-header">
                    {!showPresets && (
                        <button
                            className="toggle-presets"
                            onClick={() => setShowPresets(true)}
                        >
                            ▶ Presets
                        </button>
                    )}
                    <input
                        type="text"
                        className="shader-name-input"
                        value={shaderName}
                        onChange={(e) => setShaderName(e.target.value)}
                        placeholder="Shader name"
                    />
                    <button className="export-btn" onClick={exportShader}>
                        Export
                    </button>
                </div>

                {/* Code editor */}
                <div className="shader-editor">
                    <div className="editor-tabs">
                        <button
                            className={`tab ${activeTab === 'vertex' ? 'active' : ''}`}
                            onClick={() => setActiveTab('vertex')}
                        >
                            Vertex Shader
                        </button>
                        <button
                            className={`tab ${activeTab === 'fragment' ? 'active' : ''}`}
                            onClick={() => setActiveTab('fragment')}
                        >
                            Fragment Shader
                        </button>
                    </div>

                    <div className="editor-content">
                        {activeTab === 'vertex' ? (
                            <textarea
                                className="code-editor"
                                value={vertexShader}
                                onChange={(e) => setVertexShader(e.target.value)}
                                spellCheck={false}
                            />
                        ) : (
                            <textarea
                                className="code-editor"
                                value={fragmentShader}
                                onChange={(e) => setFragmentShader(e.target.value)}
                                spellCheck={false}
                            />
                        )}
                    </div>
                </div>

                {/* Preview & Uniforms */}
                <div className="shader-sidebar">
                    {/* Preview */}
                    <div className="shader-preview">
                        <div className="preview-header">Preview</div>
                        <canvas
                            ref={previewCanvasRef}
                            className="preview-canvas"
                            width={256}
                            height={256}
                        />
                    </div>

                    {/* Uniforms */}
                    {uniforms && (
                        <div className="shader-uniforms">
                            <div className="uniforms-header">Uniforms</div>
                            <div className="uniforms-content">
                                {Object.entries(uniforms).map(([name, def]) => (
                                    <div key={name} className="uniform-item">
                                        <label className="uniform-label">{name}</label>
                                        <div className="uniform-input">
                                            {def.type === 'color' && Array.isArray(def.value) ? (
                                                <input
                                                    type="color"
                                                    value={`#${(
                                                        (def.value[0] * 255) |
                                                        (def.value[1] * 255) << 8 |
                                                        (def.value[2] * 255) << 16
                                                    ).toString(16).padStart(6, '0')}`}
                                                    onChange={(e) => {
                                                        const hex = e.target.value;
                                                        const r = parseInt(hex.substr(1, 2), 16) / 255;
                                                        const g = parseInt(hex.substr(3, 2), 16) / 255;
                                                        const b = parseInt(hex.substr(5, 2), 16) / 255;
                                                        updateUniform(name, [r, g, b]);
                                                    }}
                                                />
                                            ) : def.type === 'float' ? (
                                                <input
                                                    type="number"
                                                    step="0.01"
                                                    value={def.value as number}
                                                    onChange={(e) => updateUniform(name, parseFloat(e.target.value))}
                                                />
                                            ) : (
                                                <span className="uniform-value">
                                                    {Array.isArray(def.value)
                                                        ? `[${def.value.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(', ')}]`
                                                        : String(def.value)}
                                                </span>
                                            )}
                                        </div>
                                        <span className="uniform-type">{def.type}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ShaderEditor;
