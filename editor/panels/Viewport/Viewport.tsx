import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import EditorManager from '../../core/EditorManager';
import TransformGizmoManager from '../../core/TransformGizmoManager';
import PlayModeToggle from '../../components/PlayModeToggle';
import FirstPersonMode from '../../components/FirstPersonMode';
import SelectionManager from '../../core/SelectionManager';
import './Viewport.css';

/**
 * Viewport Panel - Main 3D rendering area with Transform Gizmos
 *
 * This panel displays the Three.js + Spark.js rendered scene.
 * It handles:
 * - Three.js canvas rendering
 * - Object selection via raycasting
 * - Transform gizmos (Translate, Rotate, Scale)
 * - Camera controls
 * - Play/Edit mode toggle
 * - Integration with EditorManager
 */
interface ViewportProps {
    editorManager: EditorManager;
    onObjectSelected?: (object: any) => void;
}

const Viewport: React.FC<ViewportProps> = ({ editorManager, onObjectSelected }) => {
    const canvasContainerRef = useRef<HTMLDivElement>(null);
    const isPlayingRef = useRef(false);
    const [isInitialized, setIsInitialized] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [fps, setFps] = useState(60);
    const [objectCount, setObjectCount] = useState(0);
    const [transformMode, setTransformMode] = useState<'translate' | 'rotate' | 'scale'>('translate');

    /**
     * Handle canvas click for object selection
     * Disabled in play mode
     * Use useCallback with isPlayingRef to avoid stale closure issues
     */
    const handleCanvasClick = useCallback((event: MouseEvent) => {
        // Disable selection in play mode - use ref to get latest value
        if (isPlayingRef.current) return;
        if (!editorManager.canvas) return;

        const rect = editorManager.canvas.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(x, y), editorManager.camera);

        // Collect only raycastable objects (exclude splat meshes which don't support raycasting)
        const raycastableObjects: THREE.Object3D[] = [];
        editorManager.scene.traverse((obj) => {
            if (!obj.visible) return;

            const name = obj.name || '';
            const type = obj.type || '';
            const constructorName = obj.constructor.name || '';

            // Skip helpers, gizmos, grids
            if (name.includes('Grid') ||
                name.includes('Axes') ||
                name.includes('Helper') ||
                name.includes('Gizmo') ||
                name.includes('TransformControls')) {
                return;
            }

            // Skip Spark.js SplatMesh - they don't support raycasting and cause crashes
            if (type.includes('SplatMesh') ||
                constructorName.includes('Splat') ||
                name.toLowerCase().includes('splat')) {
                return;
            }

            // Only include Mesh objects
            if (obj instanceof THREE.Mesh) {
                raycastableObjects.push(obj);
            }
        });

        const intersects = raycaster.intersectObjects(raycastableObjects, true);

        if (intersects.length > 0) {
            const selectedObject = intersects[0]?.object ?? null;
            editorManager.selectObject(selectedObject);
            if (onObjectSelected) {
                onObjectSelected(selectedObject);
            }
        } else {
            editorManager.selectObject(null);
            if (onObjectSelected) {
                onObjectSelected(null);
            }
        }
    }, [editorManager, onObjectSelected]);

    useEffect(() => {
        if (!canvasContainerRef.current || !editorManager) return;

        let frameCount = 0;
        let lastTime = performance.now();

        // Initialize EditorManager with this container
        const initializeViewport = async () => {
            try {
                await editorManager.initialize(canvasContainerRef.current!);
                setIsInitialized(true);

                // Initialize SelectionManager
                const selectionManager = SelectionManager.getInstance();
                selectionManager.initialize(editorManager.scene);

                // Initialize TransformGizmoManager
                const gizmoManager = TransformGizmoManager.getInstance();
                gizmoManager.initialize(editorManager);

                // Listen for selection changes from EditorManager
                editorManager.on('selectionChanged', (data: any) => {
                    // Update selection manager
                    selectionManager.select(data.current);
                    // Notify parent component
                    if (onObjectSelected) {
                        onObjectSelected(data.current);
                    }
                });

                // Listen for gizmo mode changes
                gizmoManager.on('modeChanged', (mode: 'translate' | 'rotate' | 'scale') => {
                    setTransformMode(mode);
                });

                // Listen for play mode changes
                editorManager.on('playModeEntered', () => {
                    setIsPlaying(true);
                    isPlayingRef.current = true;
                    // Hide gizmos in play mode
                    gizmoManager.setVisible(false);
                });

                editorManager.on('playModeExited', () => {
                    setIsPlaying(false);
                    isPlayingRef.current = false;
                    // Show gizmos in edit mode
                    gizmoManager.setVisible(true);
                });

                // Setup raycasting for selection
                const canvas = canvasContainerRef.current!.querySelector('canvas');
                if (canvas) {
                    canvas.addEventListener('click', handleCanvasClick);
                }

                // Start FPS counter
                const updateFps = () => {
                    frameCount++;
                    const currentTime = performance.now();
                    if (currentTime >= lastTime + 1000) {
                        setFps(Math.round((frameCount * 1000) / (currentTime - lastTime)));
                        frameCount = 0;
                        lastTime = currentTime;

                        // Update object count
                        setObjectCount(editorManager.scene.children.length);
                    }
                    requestAnimationFrame(updateFps);
                };
                updateFps();

                console.log('Viewport: Initialized successfully with gizmos and selection');
            } catch (error) {
                console.error('Viewport: Initialization failed', error);
            }
        };

        initializeViewport();

        // Cleanup
        return () => {
            const canvas = canvasContainerRef.current?.querySelector('canvas');
            if (canvas) {
                canvas.removeEventListener('click', handleCanvasClick);
            }
        };
    }, [editorManager, onObjectSelected, handleCanvasClick]);

    /**
     * Create primitive objects
     */
    const createPrimitive = (type: 'box' | 'sphere' | 'plane' | 'cone' | 'cylinder') => {
        const object = editorManager.createPrimitive(type);
        if (onObjectSelected) {
            onObjectSelected(object);
        }
    };

    /**
     * Set transform gizmo mode
     */
    const setGizmoMode = (mode: 'translate' | 'rotate' | 'scale') => {
        const gizmoManager = TransformGizmoManager.getInstance();
        gizmoManager.setMode(mode);
        setTransformMode(mode);
    };

    /**
     * Get mode icon
     */
    const getModeIcon = (mode: 'translate' | 'rotate' | 'scale') => {
        switch (mode) {
            case 'translate':
                return '↔️';
            case 'rotate':
                return '🔄';
            case 'scale':
                return '⤡';
        }
    };

    return (
        <div className="viewport-container">
            {/* Toolbar */}
            <div className="viewport-toolbar">
                {/* Play Mode Toggle */}
                <PlayModeToggle editorManager={editorManager} onModeChanged={setIsPlaying} />

                {/* First Person Mode Toggle */}
                <FirstPersonMode editorManager={editorManager} />

                <div className="toolbar-spacer" />

                {/* Create Primitives - hidden in play mode */}
                {!isPlaying && (
                    <div className="toolbar-group">
                        <button
                            className="toolbar-button"
                            onClick={() => createPrimitive('box')}
                            title="Create Box"
                        >
                            ▢
                        </button>
                        <button
                            className="toolbar-button"
                            onClick={() => createPrimitive('sphere')}
                            title="Create Sphere"
                        >
                            ○
                        </button>
                        <button
                            className="toolbar-button"
                            onClick={() => createPrimitive('plane')}
                            title="Create Plane"
                        >
                            ▭
                        </button>
                        <button
                            className="toolbar-button"
                            onClick={() => createPrimitive('cone')}
                            title="Create Cone"
                        >
                            △
                        </button>
                        <button
                            className="toolbar-button"
                            onClick={() => createPrimitive('cylinder')}
                            title="Create Cylinder"
                        >
                            ⬭
                        </button>
                    </div>
                )}

                <div className="toolbar-spacer" />

                {/* Transform Modes - hidden in play mode */}
                {!isPlaying && (
                    <div className="toolbar-group">
                        <button
                            className={`toolbar-button ${transformMode === 'translate' ? 'active' : ''}`}
                            onClick={() => setGizmoMode('translate')}
                            title="Translate (G)"
                        >
                            ↔️
                        </button>
                        <button
                            className={`toolbar-button ${transformMode === 'rotate' ? 'active' : ''}`}
                            onClick={() => setGizmoMode('rotate')}
                            title="Rotate (R)"
                        >
                            🔄
                        </button>
                        <button
                            className={`toolbar-button ${transformMode === 'scale' ? 'active' : ''}`}
                            onClick={() => setGizmoMode('scale')}
                            title="Scale (S)"
                        >
                            ⤡
                        </button>
                    </div>
                )}

                <div className="toolbar-spacer" />

                {/* Stats */}
                <div className="viewport-stats">
                    FPS: {fps} | Objects: {objectCount}
                </div>
            </div>

            {/* Canvas container */}
            <div
                ref={canvasContainerRef}
                className="viewport-canvas"
            >
                {!isInitialized && (
                    <div className="viewport-loading">
                        <div className="loading-spinner" />
                        <p>Initializing Three.js...</p>
                    </div>
                )}
            </div>

            {/* Mode indicator */}
            {isInitialized && !isPlaying && (
                <div className="viewport-mode-indicator">
                    <span className="mode-icon">{getModeIcon(transformMode)}</span>
                    <span className="mode-text">{transformMode}</span>
                </div>
            )}

            {/* Instructions */}
            {isInitialized && (
                <div className="viewport-instructions">
                    {isPlaying ? (
                        <p>Play Mode - Press Ctrl+P or Space to exit</p>
                    ) : (
                        <p>Click to select • G/R/S for gizmos • Drag gizmo handles to transform</p>
                    )}
                </div>
            )}
        </div>
    );
};

export default Viewport;
