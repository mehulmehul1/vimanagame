import React, { useEffect, useState, useCallback } from 'react';
import EditorManager from './core/EditorManager';
import DataManager from './core/DataManager';
import AutoSaveManager from './core/AutoSaveManager';
import Viewport from './panels/Viewport/Viewport';
import Hierarchy from './panels/Hierarchy/Hierarchy';
import Inspector from './panels/Inspector/Inspector';
import SceneFlowNavigator from './panels/SceneFlowNavigator';
import { Timeline } from './panels/Timeline';
import { Console } from './panels/Console';
import { NodeGraph } from './panels/NodeGraph';
import { ShaderEditor } from './panels/ShaderEditor';
import HotReloadNotification from './components/HotReloadNotification';
import AutoSaveIndicator from './components/AutoSaveIndicator';
import LoadShadowCzarScene from './components/LoadShadowCzarScene';
import './styles/editor.css';

/**
 * Shadow Web Editor - Main Application Component
 *
 * This is the root component for the Shadow Czar Engine web editor.
 * It provides a professional Godot-like editing experience.
 *
 * Phase 1-4: Core editor features (viewport, hierarchy, inspector, scene flow)
 * Phase 5: Performance optimization (LOD, frustum culling, quality settings)
 * Phase 6: Timeline and animation
 * Phase 7: Polish features (console, node graph, shader editor, shortcuts)
 */
type CenterPanelType = 'viewport' | 'sceneflow' | 'timeline' | 'nodegraph' | 'shadereditor';
type BottomPanelType = 'none' | 'console' | 'assetbrowser';

const App: React.FC = () => {
    // Managers (singletons)
    const [editorManager] = useState<EditorManager>(() => EditorManager.getInstance());
    const [dataManager] = useState<DataManager>(() => DataManager.getInstance());
    const [autoSaveManager] = useState<AutoSaveManager>(() => AutoSaveManager.getInstance());

    // UI State
    const [isInitialized, setIsInitialized] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedObject, setSelectedObject] = useState<any>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [centerPanel, setCenterPanel] = useState<CenterPanelType>('viewport');
    const [bottomPanel, setBottomPanel] = useState<BottomPanelType>('console');

    // TODO: Get current state value from URL or game state
    const currentStateValue = 0;

    // Panel toggle handlers
    const toggleCenterPanel = useCallback((panel: CenterPanelType) => {
        setCenterPanel(prev => prev === panel ? 'viewport' : panel);
    }, []);

    const toggleBottomPanel = useCallback((panel: BottomPanelType) => {
        setBottomPanel(prev => prev === panel ? 'none' : panel);
    }, []);

    useEffect(() => {
        // Initialize editor systems
        const initializeEditor = async () => {
            try {
                console.log('Shadow Web Editor initializing...');

                // Initialize AutoSaveManager
                console.log('Initializing AutoSaveManager...');
                autoSaveManager.enable(2);

                // EditorManager will be initialized by Viewport component
                setIsInitialized(true);

                console.log('Phase 5-7: Performance, Animation, Polish complete');
                console.log('- LOD Manager, Frustum Culling, Quality Settings');
                console.log('- Timeline with keyframe animation');
                console.log('- Console/Profiler panel');
                console.log('- Node Graph for visual scripting');
                console.log('- Shader Editor for TSL shaders');
                console.log('- Keyboard shortcuts system');
                console.log('- User preferences with localStorage');

                // Listen to selection changes
                editorManager.on('selectionChanged', (data: any) => {
                    setSelectedObject(data.current);
                });

                // Listen to play mode changes
                editorManager.on('playModeEntered', () => setIsPlaying(true));
                editorManager.on('playModeExited', () => setIsPlaying(false));

            } catch (err) {
                const errorMessage = err instanceof Error ? err.message : 'Unknown error';
                setError(errorMessage);
                console.error('Failed to initialize editor:', err);
            }
        };

        initializeEditor();

        // Setup keyboard shortcuts
        const handleKeyDown = (event: KeyboardEvent) => {
            // Ignore if typing in input
            if (event.target instanceof HTMLInputElement ||
                event.target instanceof HTMLTextAreaElement) {
                return;
            }

            switch (event.key) {
                case 'F4':
                    event.preventDefault();
                    toggleCenterPanel('sceneflow');
                    break;
                case 'F5':
                    event.preventDefault();
                    toggleCenterPanel('timeline');
                    break;
                case 'F6':
                    event.preventDefault();
                    toggleCenterPanel('nodegraph');
                    break;
                case 'F7':
                    event.preventDefault();
                    toggleCenterPanel('shadereditor');
                    break;
                case 'F8':
                    event.preventDefault();
                    toggleBottomPanel('console');
                    break;
                case 'F1':
                    event.preventDefault();
                    alert('Keyboard Shortcuts:\nG - Translate Gizmo\nR - Rotate Gizmo\nS - Scale Gizmo\nDel - Delete Object\nCtrl+D - Duplicate\nF2 - Rename\nF4 - Scene Flow\nF5 - Timeline\nF6 - Node Graph\nF7 - Shader Editor\nF8 - Console');
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        // Cleanup
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            autoSaveManager.destroy();
            editorManager.destroy();
        };
    }, [editorManager, autoSaveManager, toggleCenterPanel, toggleBottomPanel]);

    if (error) {
        return (
            <div className="error-screen">
                <h1>Shadow Web Editor - Initialization Error</h1>
                <pre>{error}</pre>
            </div>
        );
    }

    if (!isInitialized) {
        return (
            <div className="loading-screen">
                <div className="loading-spinner" />
                <h1>Shadow Web Editor</h1>
                <p>Loading editor systems...</p>
            </div>
        );
    }

    // Render center panel based on selection
    const renderCenterPanel = () => {
        switch (centerPanel) {
            case 'sceneflow':
                return (
                    <div className="panel-scene-flow">
                        <SceneFlowNavigator
                            currentStateValue={currentStateValue}
                            onJumpToState={(stateName) => {
                                const url = new URL(window.location.href);
                                url.searchParams.set('gameState', stateName);
                                window.location.href = url.toString();
                            }}
                        />
                    </div>
                );
            case 'timeline':
                return (
                    <div className="panel-timeline">
                        <Timeline editorManager={editorManager} />
                    </div>
                );
            case 'nodegraph':
                return (
                    <div className="panel-nodegraph">
                        <NodeGraph editorManager={editorManager} />
                    </div>
                );
            case 'shadereditor':
                return (
                    <div className="panel-shadereditor">
                        <ShaderEditor editorManager={editorManager} />
                    </div>
                );
            default:
                return (
                    <div className="panel-viewport">
                        <Viewport
                            editorManager={editorManager}
                            onObjectSelected={setSelectedObject}
                        />
                    </div>
                );
        }
    };

    // Render bottom panel based on selection
    const renderBottomPanel = () => {
        if (bottomPanel === 'none') return null;

        switch (bottomPanel) {
            case 'console':
                return (
                    <div className="panel-console">
                        <Console editorManager={editorManager} />
                    </div>
                );
            case 'assetbrowser':
                return null; // Asset browser coming soon
            default:
                return null;
        }
    };

    return (
        <div className="editor-container">
            {/* Top Menu Bar */}
            <div className="editor-menubar">
                <div className="menubar-left">
                    <span className="editor-title">🎮 Shadow Web Editor</span>
                    <span className="menubar-separator">|</span>
                    <LoadShadowCzarScene />
                    <span className="menubar-separator">|</span>

                    {/* Panel Toggles */}
                    <button
                        className={`menubar-toggle ${centerPanel === 'viewport' ? 'active' : ''}`}
                        onClick={() => setCenterPanel('viewport')}
                        title="Viewport"
                    >
                        ◐ Viewport
                    </button>
                    <button
                        className={`menubar-toggle ${centerPanel === 'sceneflow' ? 'active' : ''}`}
                        onClick={() => toggleCenterPanel('sceneflow')}
                        title="Scene Flow Navigator (F4)"
                    >
                        ◪ Scene Flow
                    </button>
                    <button
                        className={`menubar-toggle ${centerPanel === 'timeline' ? 'active' : ''}`}
                        onClick={() => toggleCenterPanel('timeline')}
                        title="Timeline (F5)"
                    >
                        📊 Timeline
                    </button>
                    <button
                        className={`menubar-toggle ${centerPanel === 'nodegraph' ? 'active' : ''}`}
                        onClick={() => toggleCenterPanel('nodegraph')}
                        title="Node Graph (F6)"
                    >
                        🔗 Node Graph
                    </button>
                    <button
                        className={`menubar-toggle ${centerPanel === 'shadereditor' ? 'active' : ''}`}
                        onClick={() => toggleCenterPanel('shadereditor')}
                        title="Shader Editor (F7)"
                    >
                        🎨 Shader
                    </button>

                    <span className="menubar-separator">|</span>

                    {/* Bottom Panel Toggles */}
                    <button
                        className={`menubar-toggle ${bottomPanel === 'console' ? 'active' : ''}`}
                        onClick={() => toggleBottomPanel('console')}
                        title="Console (F8)"
                    >
                        ⬚ Console
                    </button>

                    <span className="menubar-separator">|</span>
                    <span className="menubar-info">Phase 5-7 Complete</span>
                </div>
                <div className="menubar-right">
                    <AutoSaveIndicator autoSaveManager={autoSaveManager} />
                </div>
            </div>

            {/* Hot Reload Notification */}
            <HotReloadNotification dataManager={dataManager} />

            {/* Main Editor Area */}
            <div className="editor-main">
                {/* Left Panel - Hierarchy */}
                <div className="panel-hierarchy">
                    <Hierarchy
                        editorManager={editorManager}
                        selectedObject={selectedObject}
                        onObjectSelected={setSelectedObject}
                    />
                </div>

                {/* Center Panel - Dynamic */}
                {renderCenterPanel()}

                {/* Right Panel - Inspector */}
                <div className="panel-inspector">
                    <Inspector
                        editorManager={editorManager}
                        selectedObject={selectedObject}
                    />
                </div>
            </div>

            {/* Bottom Panel - Dynamic */}
            {renderBottomPanel()}

            {/* Bottom Status Bar */}
            <div className="editor-statusbar">
                <span className="status-text">{isPlaying ? '▶ Playing...' : '▌▌ Ready'}</span>
                <span className="status-info">
                    {selectedObject ? `Selected: ${selectedObject.name || 'Unnamed'} (${selectedObject.type})` : 'No selection'}
                </span>
                <span className="status-quality">
                    FPS: <span id="fps-counter">60</span>
                </span>
            </div>
        </div>
    );
};

export default App;
