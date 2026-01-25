import React, { useState, useRef, useEffect } from 'react';
import EditorManager from '../core/EditorManager';
import DataManager from '../core/DataManager';
import UndoRedoManager from '../core/UndoRedoManager';
import './MenuBar.css';

/**
 * MenuBar - Enhanced menubar with dropdown menus
 *
 * Features:
 * - File: New, Open, Save, Export, Exit
 * - Edit: Undo, Redo, Cut, Copy, Paste, Duplicate, Delete
 * - View: Panel toggles, Fullscreen, Stats overlay
 * - Help: Documentation, Shortcuts, About
 */
interface MenuBarProps {
    editorManager: EditorManager;
    dataManager: DataManager;
    onTogglePanel?: (panel: 'hierarchy' | 'inspector' | 'timeline' | 'console' | 'nodegraph' | 'shadereditor') => void;
    onNewScene?: () => void;
}

const MenuBar: React.FC<MenuBarProps> = ({
    editorManager,
    dataManager,
    onTogglePanel,
    onNewScene
}) => {
    const [openMenu, setOpenMenu] = useState<string | null>(null);
    const [canUndo, setCanUndo] = useState(false);
    const [canRedo, setCanRedo] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);

    const undoRedoManager = UndoRedoManager.getInstance();

    // Update undo/redo state
    useEffect(() => {
        const updateState = () => {
            setCanUndo(undoRedoManager.canUndo());
            setCanRedo(undoRedoManager.canRedo());
        };

        updateState();
        const interval = setInterval(updateState, 500);

        return () => clearInterval(interval);
    }, [undoRedoManager]);

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setOpenMenu(null);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    /**
     * Handle menu item click
     */
    const handleMenuAction = async (menu: string, action: string) => {
        setOpenMenu(null);

        switch (menu) {
            case 'file':
                switch (action) {
                    case 'new':
                        if (onNewScene) {
                            if (confirm('Create a new scene? Unsaved changes will be lost.')) {
                                onNewScene();
                            }
                        }
                        break;
                    case 'open':
                        // Trigger open dialog
                        const input = document.createElement('input');
                        input.type = 'file';
                        input.accept = '.js,.json';
                        input.onchange = async (e) => {
                            const file = (e.target as HTMLInputElement).files?.[0];
                            if (file) {
                                const text = await file.text();
                                try {
                                    const data = JSON.parse(text);
                                    console.log('Loaded scene data:', data);
                                    // Emit event for DataManager to handle
                                    dataManager.emit('importScene', data);
                                } catch (err) {
                                    alert('Failed to parse scene file');
                                }
                            }
                        };
                        input.click();
                        break;
                    case 'save':
                        // Trigger save
                        dataManager.emit('saveRequested');
                        break;
                    case 'export':
                        // Export scene to JSON
                        const sceneData = dataManager.getSceneData();
                        if (sceneData) {
                            const blob = new Blob([JSON.stringify(sceneData, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'scene.json';
                            a.click();
                            URL.revokeObjectURL(url);
                        }
                        break;
                    case 'exit':
                        if (confirm('Exit the editor?')) {
                            window.close();
                        }
                        break;
                }
                break;

            case 'edit':
                switch (action) {
                    case 'undo':
                        if (canUndo) undoRedoManager.undo();
                        break;
                    case 'redo':
                        if (canRedo) undoRedoManager.redo();
                        break;
                    case 'duplicate':
                        dataManager.emit('duplicateRequested');
                        break;
                    case 'delete':
                        dataManager.emit('deleteRequested');
                        break;
                }
                break;

            case 'view':
                switch (action) {
                    case 'fullscreen':
                        if (document.fullscreenElement) {
                            document.exitFullscreen();
                        } else {
                            document.documentElement.requestFullscreen();
                        }
                        break;
                    case 'stats':
                        // Toggle stats overlay
                        const stats = document.querySelector('.stats-overlay');
                        if (stats) {
                            stats.classList.toggle('hidden');
                        }
                        break;
                    default:
                        if (action.startsWith('panel-') && onTogglePanel) {
                            const panel = action.replace('panel-', '') as any;
                            onTogglePanel(panel);
                        }
                }
                break;

            case 'help':
                switch (action) {
                    case 'docs':
                        window.open('/docs/editor-user-guide.html', '_blank');
                        break;
                    case 'shortcuts':
                        // Show shortcuts modal
                        setOpenMenu('shortcuts-modal');
                        break;
                    case 'about':
                        alert('Shadow Web Editor v1.0\n\nA professional web-based 3D game editor for the Shadow Czar Engine.');
                        break;
                }
                break;
        }
    };

    /**
     * Get keyboard shortcut for action
     */
    const getShortcut = (action: string): string => {
        const shortcuts: Record<string, string> = {
            'save': 'Ctrl+S',
            'undo': 'Ctrl+Z',
            'redo': 'Ctrl+Shift+Z',
            'duplicate': 'Ctrl+D',
            'delete': 'Del',
            'new': 'Ctrl+N',
            'open': 'Ctrl+O',
            'export': 'Ctrl+E',
            'shortcuts': 'F1',
        };
        return shortcuts[action] || '';
    };

    /**
     * Render menu dropdown
     */
    const renderMenu = (menuName: string, items: Array<{
        label: string;
        action: string;
        shortcut?: string;
        separator?: boolean;
        disabled?: boolean;
    }>) => (
        <div
            key={menuName}
            className={`menu-dropdown ${openMenu === menuName ? 'open' : ''}`}
            onClick={(e) => e.stopPropagation()}
        >
            {items.map((item, index) => (
                <React.Fragment key={index}>
                    {item.separator ? (
                        <div className="menu-separator" />
                    ) : (
                        <button
                            className={`menu-item ${item.disabled ? 'disabled' : ''}`}
                            onClick={() => !item.disabled && handleMenuAction(menuName, item.action)}
                            disabled={item.disabled}
                        >
                            <span className="menu-item-label">{item.label}</span>
                            {(item.shortcut || getShortcut(item.action)) && (
                                <span className="menu-item-shortcut">
                                    {item.shortcut || getShortcut(item.action)}
                                </span>
                            )}
                        </button>
                    )}
                </React.Fragment>
            ))}
        </div>
    );

    /**
     * Menu definitions
     */
    const menus = {
        file: [
            { label: 'New Scene', action: 'new', shortcut: 'Ctrl+N' },
            { label: 'Open Scene...', action: 'open', shortcut: 'Ctrl+O' },
            { separator: true },
            { label: 'Save Scene', action: 'save', shortcut: 'Ctrl+S' },
            { label: 'Export Scene...', action: 'export', shortcut: 'Ctrl+E' },
            { separator: true },
            { label: 'Exit', action: 'exit' },
        ],
        edit: [
            { label: 'Undo', action: 'undo', shortcut: 'Ctrl+Z', disabled: !canUndo },
            { label: 'Redo', action: 'redo', shortcut: 'Ctrl+Shift+Z', disabled: !canRedo },
            { separator: true },
            { label: 'Duplicate', action: 'duplicate', shortcut: 'Ctrl+D' },
            { label: 'Delete', action: 'delete', shortcut: 'Del' },
        ],
        view: [
            { label: 'Hierarchy', action: 'panel-hierarchy' },
            { label: 'Inspector', action: 'panel-inspector' },
            { label: 'Timeline', action: 'panel-timeline' },
            { label: 'Console', action: 'panel-console' },
            { label: 'Node Graph', action: 'panel-nodegraph' },
            { label: 'Shader Editor', action: 'panel-shadereditor' },
            { separator: true },
            { label: 'Fullscreen', action: 'fullscreen' },
            { label: 'Stats Overlay', action: 'stats' },
        ],
        help: [
            { label: 'Documentation', action: 'docs' },
            { label: 'Keyboard Shortcuts', action: 'shortcuts', shortcut: 'F1' },
            { separator: true },
            { label: 'About', action: 'about' },
        ]
    };

    return (
        <div className="menubar-container" ref={menuRef}>
            <div className="menubar">
                {/* Logo */}
                <div className="menubar-logo">
                    <span className="logo-icon">◆</span>
                    <span className="logo-text">Shadow Editor</span>
                </div>

                {/* Menu items */}
                {Object.entries(menus).map(([name, items]) => (
                    <div
                        key={name}
                        className={`menubar-item ${openMenu === name ? 'active' : ''}`}
                        onClick={() => setOpenMenu(openMenu === name ? null : name)}
                    >
                        {name.charAt(0).toUpperCase() + name.slice(1)}
                    </div>
                ))}

                {/* Spacer */}
                <div className="menubar-spacer" />

                {/* Toolbar actions */}
                <div className="menubar-toolbar">
                    <button
                        className="toolbar-btn"
                        onClick={() => handleMenuAction('edit', 'undo')}
                        disabled={!canUndo}
                        title="Undo (Ctrl+Z)"
                    >
                        ↶
                    </button>
                    <button
                        className="toolbar-btn"
                        onClick={() => handleMenuAction('edit', 'redo')}
                        disabled={!canRedo}
                        title="Redo (Ctrl+Shift+Z)"
                    >
                        ↷
                    </button>
                    <div className="toolbar-divider" />
                    <button
                        className="toolbar-btn"
                        onClick={() => handleMenuAction('file', 'save')}
                        title="Save (Ctrl+S)"
                    >
                        💾
                    </button>
                    <button
                        className="toolbar-btn"
                        onClick={() => editorManager.isPlaying ? editorManager.exitPlayMode() : editorManager.enterPlayMode()}
                        title={editorManager.isPlaying ? 'Stop (Ctrl+P)' : 'Play (Ctrl+P)'}
                    >
                        {editorManager.isPlaying ? '⏹' : '▶'}
                    </button>
                </div>

                {/* Menus */}
                {Object.entries(menus).map(([name, items]) => renderMenu(name, items))}
            </div>

            {/* Shortcuts modal */}
            {openMenu === 'shortcuts-modal' && (
                <div className="modal-overlay" onClick={() => setOpenMenu(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>Keyboard Shortcuts</h3>
                            <button className="modal-close" onClick={() => setOpenMenu(null)}>×</button>
                        </div>
                        <div className="modal-body">
                            <div className="shortcuts-grid">
                                <div className="shortcut-category">
                                    <h4>File</h4>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+N</span>
                                        <span className="shortcut-desc">New Scene</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+O</span>
                                        <span className="shortcut-desc">Open Scene</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+S</span>
                                        <span className="shortcut-desc">Save Scene</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+E</span>
                                        <span className="shortcut-desc">Export Scene</span>
                                    </div>
                                </div>
                                <div className="shortcut-category">
                                    <h4>Edit</h4>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+Z</span>
                                        <span className="shortcut-desc">Undo</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+Shift+Z</span>
                                        <span className="shortcut-desc">Redo</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+D</span>
                                        <span className="shortcut-desc">Duplicate</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Del</span>
                                        <span className="shortcut-desc">Delete</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">F2</span>
                                        <span className="shortcut-desc">Rename</span>
                                    </div>
                                </div>
                                <div className="shortcut-category">
                                    <h4>Viewport</h4>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">G</span>
                                        <span className="shortcut-desc">Translate</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">R</span>
                                        <span className="shortcut-desc">Rotate</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">S</span>
                                        <span className="shortcut-desc">Scale</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">F</span>
                                        <span className="shortcut-desc">Focus</span>
                                    </div>
                                </div>
                                <div className="shortcut-category">
                                    <h4>Panels</h4>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+1</span>
                                        <span className="shortcut-desc">Hierarchy</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+2</span>
                                        <span className="shortcut-desc">Inspector</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+3</span>
                                        <span className="shortcut-desc">Timeline</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+4</span>
                                        <span className="shortcut-desc">Console</span>
                                    </div>
                                </div>
                                <div className="shortcut-category">
                                    <h4>Play</h4>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Ctrl+P</span>
                                        <span className="shortcut-desc">Toggle Play/Edit</span>
                                    </div>
                                    <div className="shortcut-item">
                                        <span className="shortcut-key">Space</span>
                                        <span className="shortcut-desc">Play/Pause Timeline</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MenuBar;
