import React, { useState, useEffect, useRef, useCallback } from 'react';
import EditorManager from '../../core/EditorManager';
import './Inspector.css';

/**
 * Inspector Panel - Enhanced object properties editor
 *
 * Sections:
 * - Transform (Position, Rotation, Scale)
 * - Material (Color, Metalness, Roughness, etc.)
 * - Criteria (State, Flags - for runtime)
 * - Custom Properties (User-defined key-value pairs)
 * - Real-time updates when gizmo moves object
 */
interface InspectorProps {
    editorManager: EditorManager;
    selectedObject: any;
}

interface MaterialData {
    color: string;
    metalness: number;
    roughness: number;
    opacity: number;
    emissive: string;
}

interface CriteriaData {
    state: string[];
    flags: string[];
}

interface CustomProperty {
    key: string;
    value: any;
    type: 'string' | 'number' | 'boolean';
}

const Inspector: React.FC<InspectorProps> = ({ editorManager, selectedObject }) => {
    // Transform state
    const [position, setPosition] = useState({ x: 0, y: 0, z: 0 });
    const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
    const [scale, setScale] = useState({ x: 1, y: 1, z: 1 });

    // Material state
    const [material, setMaterial] = useState<MaterialData>({
        color: '#ffffff',
        metalness: 0,
        roughness: 0.5,
        opacity: 1,
        emissive: '#000000'
    });

    // Criteria state
    const [criteria, setCriteria] = useState<CriteriaData>({
        state: [],
        flags: []
    });

    // Custom properties
    const [customProps, setCustomProps] = useState<CustomProperty[]>([]);
    const [newPropKey, setNewPropKey] = useState('');
    const [newPropValue, setNewPropValue] = useState('');
    const [newPropType, setNewPropType] = useState<'string' | 'number' | 'boolean'>('string');

    // Sections collapsed state
    const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());

    // Track if user is currently editing (to prevent overwriting input)
    const isEditingRef = useRef(false);

    // Update all values from selected object
    const updateAllValues = useCallback(() => {
        if (!selectedObject || isEditingRef.current) return;

        // Update transform
        if (selectedObject.position) {
            setPosition({
                x: parseFloat(selectedObject.position.x.toFixed(3)),
                y: parseFloat(selectedObject.position.y.toFixed(3)),
                z: parseFloat(selectedObject.position.z.toFixed(3))
            });
        }

        if (selectedObject.rotation) {
            setRotation({
                x: parseFloat(((selectedObject.rotation.x * 180) / Math.PI).toFixed(1)),
                y: parseFloat(((selectedObject.rotation.y * 180) / Math.PI).toFixed(1)),
                z: parseFloat(((selectedObject.rotation.z * 180) / Math.PI).toFixed(1))
            });
        }

        if (selectedObject.scale) {
            setScale({
                x: parseFloat(selectedObject.scale.x.toFixed(3)),
                y: parseFloat(selectedObject.scale.y.toFixed(3)),
                z: parseFloat(selectedObject.scale.z.toFixed(3))
            });
        }

        // Update material
        if (selectedObject.material) {
            const mat = selectedObject.material;
            setMaterial({
                color: '#' + (mat.color?.getHexString() || 'ffffff'),
                metalness: (mat as any).metalness || 0,
                roughness: (mat as any).roughness || 0.5,
                opacity: (mat as any).opacity || 1,
                emissive: '#' + ((mat as any).emissive?.getHexString() || '000000')
            });
        }

        // Update criteria
        if (selectedObject.userData?.criteria) {
            setCriteria(selectedObject.userData.criteria);
        }

        // Update custom properties
        if (selectedObject.userData?.customProperties) {
            setCustomProps(selectedObject.userData.customProperties);
        }
    }, [selectedObject]);

    // Set up event listeners for gizmo changes
    useEffect(() => {
        if (!editorManager) return;

        // Listen for selection changes to refresh inspector
        const handleSelectionChanged = () => {
            updateAllValues();
        };

        // Listen for transform changes from gizmo
        const handleTransformChanged = () => {
            updateAllValues();
        };

        editorManager.on('selectionChanged', handleSelectionChanged);
        editorManager.on('transformChanged', handleTransformChanged);

        return () => {
            editorManager.off('selectionChanged', handleSelectionChanged);
            editorManager.off('transformChanged', handleTransformChanged);
        };
    }, [editorManager, updateAllValues]);

    // Update when selection changes
    useEffect(() => {
        updateAllValues();
    }, [selectedObject, updateAllValues]);

    const toggleSection = (section: string) => {
        const newCollapsed = new Set(collapsedSections);
        if (newCollapsed.has(section)) {
            newCollapsed.delete(section);
        } else {
            newCollapsed.add(section);
        }
        setCollapsedSections(newCollapsed);
    };

    const handlePositionChange = (axis: 'x' | 'y' | 'z', value: number) => {
        if (!selectedObject) return;
        selectedObject.position[axis] = value;
        setPosition(prev => ({ ...prev, [axis]: value }));
    };

    const handleRotationChange = (axis: 'x' | 'y' | 'z', value: number) => {
        if (!selectedObject) return;
        selectedObject.rotation[axis] = (value * Math.PI) / 180;
        setRotation(prev => ({ ...prev, [axis]: value }));
    };

    const handleScaleChange = (axis: 'x' | 'y' | 'z', value: number) => {
        if (!selectedObject) return;
        selectedObject.scale[axis] = value;
        setScale(prev => ({ ...prev, [axis]: value }));
    };

    const handleMaterialChange = (key: keyof MaterialData, value: any) => {
        if (!selectedObject?.material) return;

        setMaterial(prev => ({ ...prev, [key]: value }));

        const mat = selectedObject.material;
        switch (key) {
            case 'color':
            case 'emissive':
                mat[key].set(value);
                break;
            case 'metalness':
            case 'roughness':
            case 'opacity':
                mat[key] = value;
                break;
        }
    };

    const handleCriteriaChange = (type: 'state' | 'flags', value: string) => {
        if (!selectedObject) return;

        const items = value.split(',').map(s => s.trim()).filter(s => s);
        const newCriteria = { ...criteria, [type]: items };
        setCriteria(newCriteria);

        selectedObject.userData = selectedObject.userData || {};
        selectedObject.userData.criteria = newCriteria;
    };

    const addCustomProperty = () => {
        if (!selectedObject || !newPropKey.trim()) return;

        let value: any = newPropValue;
        if (newPropType === 'number') {
            value = parseFloat(newPropValue) || 0;
        } else if (newPropType === 'boolean') {
            value = newPropValue.toLowerCase() === 'true';
        }

        const newProp: CustomProperty = {
            key: newPropKey.trim(),
            value,
            type: newPropType
        };

        const updatedProps = [...customProps, newProp];
        setCustomProps(updatedProps);

        selectedObject.userData = selectedObject.userData || {};
        selectedObject.userData.customProperties = updatedProps;

        setNewPropKey('');
        setNewPropValue('');
        setNewPropType('string');
    };

    const removeCustomProperty = (index: number) => {
        if (!selectedObject) return;

        const updatedProps = customProps.filter((_, i) => i !== index);
        setCustomProps(updatedProps);

        selectedObject.userData.customProperties = updatedProps;
    };

    const updateCustomProperty = (index: number, value: any) => {
        if (!selectedObject) return;

        const updatedProps = [...customProps];
        if (updatedProps[index]) {
            updatedProps[index].value = value;
        }
        setCustomProps(updatedProps);

        selectedObject.userData.customProperties = updatedProps;
    };

    const Vector3Input = ({
        label,
        values,
        onChange,
        step = 0.1
    }: {
        label: string;
        values: { x: number; y: number; z: number };
        onChange: (axis: 'x' | 'y' | 'z', value: number) => void;
        step?: number;
    }) => {
        if (!values) return null;
        return (
            <div className="inspector-section">
                <div className="inspector-section-header" onClick={() => toggleSection(label)}>
                    <span>{label}</span>
                    <span className="collapse-icon">{collapsedSections.has(label) ? '▶' : '▼'}</span>
                </div>
                {!collapsedSections.has(label) && (
                    <div className="vector3-input">
                        <div className="vector3-field">
                            <label className="axis-label x-axis">X</label>
                            <input
                                type="number"
                                value={values.x}
                                onChange={(e) => onChange('x', parseFloat(e.target.value) || 0)}
                                step={step}
                                className="inspector-input"
                            />
                        </div>
                        <div className="vector3-field">
                            <label className="axis-label y-axis">Y</label>
                            <input
                                type="number"
                                value={values.y}
                                onChange={(e) => onChange('y', parseFloat(e.target.value) || 0)}
                                step={step}
                                className="inspector-input"
                            />
                        </div>
                        <div className="vector3-field">
                            <label className="axis-label z-axis">Z</label>
                            <input
                                type="number"
                                value={values.z}
                                onChange={(e) => onChange('z', parseFloat(e.target.value) || 0)}
                                step={step}
                                className="inspector-input"
                            />
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="inspector-container">
            <div className="inspector-header">
                <h3>Inspector</h3>
            </div>

            <div className="inspector-content">
                {!selectedObject ? (
                    <div className="inspector-empty">
                        <p>No object selected</p>
                        <p className="hint">Click an object in the viewport or hierarchy to inspect</p>
                    </div>
                ) : (
                    <>
                        {/* Object Name */}
                        <div className="inspector-section">
                            <div className="inspector-section-header">Name</div>
                            <input
                                type="text"
                                value={selectedObject.name || ''}
                                onChange={(e) => {
                                    selectedObject.name = e.target.value;
                                }}
                                className="inspector-input full-width"
                                placeholder="Unnamed Object"
                            />
                        </div>

                        {/* Object Type */}
                        <div className="inspector-section">
                            <div className="inspector-section-header">Type</div>
                            <div className="object-type">
                                {selectedObject.type || 'Object3D'}
                            </div>
                        </div>

                        {/* UUID */}
                        <div className="inspector-section">
                            <div className="inspector-section-header">UUID</div>
                            <div className="object-uuid">{selectedObject.uuid.slice(0, 8)}...</div>
                        </div>

                        {/* Transform */}
                        <Vector3Input
                            label="Position"
                            values={position}
                            onChange={handlePositionChange}
                            step={0.1}
                        />

                        <Vector3Input
                            label="Rotation"
                            values={rotation}
                            onChange={handleRotationChange}
                            step={1}
                        />

                        <Vector3Input
                            label="Scale"
                            values={scale}
                            onChange={handleScaleChange}
                            step={0.1}
                        />

                        {/* Material Section */}
                        {selectedObject.type && selectedObject.type.includes('Mesh') && selectedObject.material && (
                            <div className="inspector-section">
                                <div className="inspector-section-header" onClick={() => toggleSection('Material')}>
                                    <span>Material</span>
                                    <span className="collapse-icon">{collapsedSections.has('Material') ? '▶' : '▼'}</span>
                                </div>
                                {!collapsedSections.has('Material') && (
                                    <div className="material-inputs">
                                        <div className="material-row">
                                            <label className="material-label">Color</label>
                                            <input
                                                type="color"
                                                value={material.color}
                                                onChange={(e) => handleMaterialChange('color', e.target.value)}
                                                className="color-input"
                                            />
                                            <input
                                                type="text"
                                                value={material.color}
                                                onChange={(e) => handleMaterialChange('color', e.target.value)}
                                                className="inspector-input"
                                            />
                                        </div>
                                        <div className="material-row">
                                            <label className="material-label">Metalness</label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.metalness}
                                                onChange={(e) => handleMaterialChange('metalness', parseFloat(e.target.value))}
                                                className="slider-input"
                                            />
                                            <input
                                                type="number"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.metalness}
                                                onChange={(e) => handleMaterialChange('metalness', parseFloat(e.target.value))}
                                                className="inspector-input small"
                                            />
                                        </div>
                                        <div className="material-row">
                                            <label className="material-label">Roughness</label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.roughness}
                                                onChange={(e) => handleMaterialChange('roughness', parseFloat(e.target.value))}
                                                className="slider-input"
                                            />
                                            <input
                                                type="number"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.roughness}
                                                onChange={(e) => handleMaterialChange('roughness', parseFloat(e.target.value))}
                                                className="inspector-input small"
                                            />
                                        </div>
                                        <div className="material-row">
                                            <label className="material-label">Opacity</label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.opacity}
                                                onChange={(e) => handleMaterialChange('opacity', parseFloat(e.target.value))}
                                                className="slider-input"
                                            />
                                            <input
                                                type="number"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={material.opacity}
                                                onChange={(e) => handleMaterialChange('opacity', parseFloat(e.target.value))}
                                                className="inspector-input small"
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Criteria Section */}
                        <div className="inspector-section">
                            <div className="inspector-section-header" onClick={() => toggleSection('Criteria')}>
                                <span>Criteria</span>
                                <span className="collapse-icon">{collapsedSections.has('Criteria') ? '▶' : '▼'}</span>
                            </div>
                            {!collapsedSections.has('Criteria') && (
                                <div className="criteria-inputs">
                                    <div className="criteria-row">
                                        <label className="criteria-label">State</label>
                                        <input
                                            type="text"
                                            value={criteria.state.join(', ')}
                                            onChange={(e) => handleCriteriaChange('state', e.target.value)}
                                            className="inspector-input full-width"
                                            placeholder="e.g., active, disabled, hidden"
                                        />
                                    </div>
                                    <div className="criteria-row">
                                        <label className="criteria-label">Flags</label>
                                        <input
                                            type="text"
                                            value={criteria.flags.join(', ')}
                                            onChange={(e) => handleCriteriaChange('flags', e.target.value)}
                                            className="inspector-input full-width"
                                            placeholder="e.g., interactable, collision, trigger"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Custom Properties Section */}
                        <div className="inspector-section">
                            <div className="inspector-section-header" onClick={() => toggleSection('Custom Properties')}>
                                <span>Custom Properties</span>
                                <span className="collapse-icon">{collapsedSections.has('Custom Properties') ? '▶' : '▼'}</span>
                            </div>
                            {!collapsedSections.has('Custom Properties') && (
                                <div className="custom-props">
                                    {customProps.map((prop, index) => (
                                        <div key={index} className="custom-prop-row">
                                            <input
                                                type="text"
                                                value={prop.key}
                                                readOnly
                                                className="inspector-input prop-key"
                                            />
                                            <input
                                                type={prop.type === 'number' ? 'number' : 'text'}
                                                value={prop.value}
                                                onChange={(e) => {
                                                    const value = prop.type === 'number'
                                                        ? parseFloat(e.target.value)
                                                        : e.target.value;
                                                    updateCustomProperty(index, value);
                                                }}
                                                className="inspector-input prop-value"
                                            />
                                            <button
                                                className="prop-remove-btn"
                                                onClick={() => removeCustomProperty(index)}
                                            >
                                                ×
                                            </button>
                                        </div>
                                    ))}
                                    <div className="custom-prop-add">
                                        <input
                                            type="text"
                                            value={newPropKey}
                                            onChange={(e) => setNewPropKey(e.target.value)}
                                            placeholder="Key"
                                            className="inspector-input prop-key"
                                        />
                                        <input
                                            type="text"
                                            value={newPropValue}
                                            onChange={(e) => setNewPropValue(e.target.value)}
                                            placeholder="Value"
                                            className="inspector-input prop-value"
                                        />
                                        <select
                                            value={newPropType}
                                            onChange={(e) => setNewPropType(e.target.value as any)}
                                            className="inspector-input prop-type"
                                        >
                                            <option value="string">String</option>
                                            <option value="number">Number</option>
                                            <option value="boolean">Boolean</option>
                                        </select>
                                        <button
                                            className="prop-add-btn"
                                            onClick={addCustomProperty}
                                            disabled={!newPropKey.trim()}
                                        >
                                            +
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default Inspector;
