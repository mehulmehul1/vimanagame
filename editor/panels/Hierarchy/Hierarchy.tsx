import React, { useState, useEffect, useRef } from 'react';
import EditorManager from '../../core/EditorManager';
import SceneLoader from '../../core/SceneLoader';
import SelectionManager from '../../core/SelectionManager';
import './Hierarchy.css';

/**
 * Hierarchy Panel - Enhanced scene graph tree view
 *
 * Features:
 * - Hierarchical tree with expand/collapse
 * - Inline rename (F2 or double-click)
 * - Show/hide toggle (eye icon)
 * - Lock toggle (padlock icon)
 * - Multi-selection (Ctrl+Click)
 * - Drag-drop reparenting (basic)
 * - Icons for different object types
 */
interface HierarchyProps {
    editorManager: EditorManager;
    selectedObject: any;
    onObjectSelected?: (object: any) => void;
}

interface TreeNode {
    object: any;
    children: TreeNode[];
    expanded: boolean;
    visible: boolean;
    locked: boolean;
    depth: number;
}

const Hierarchy: React.FC<HierarchyProps> = ({ editorManager, selectedObject, onObjectSelected }) => {
    const [treeNodes, setTreeNodes] = useState<TreeNode[]>([]);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editingName, setEditingName] = useState<string>('');
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!editorManager?.scene) return;

        // Get SceneLoader instance for auto-refresh
        const sceneLoader = SceneLoader.getInstance();

        // Initial load
        buildTree();

        // Update when objects are created/deleted
        const handleObjectCreated = () => buildTree();
        const handleObjectDeleted = () => buildTree();
        const handleSelectionChanged = () => buildTree();

        // Update when scene objects are loaded via SceneLoader
        const handleObjectLoaded = () => buildTree();
        const handleObjectUnloaded = () => buildTree();

        editorManager.on('objectCreated', handleObjectCreated);
        editorManager.on('objectDeleted', handleObjectDeleted);
        editorManager.on('selectionChanged', handleSelectionChanged);

        // Listen to SceneLoader events for auto-refresh
        sceneLoader.on('objectLoaded', handleObjectLoaded);
        sceneLoader.on('objectUnloaded', handleObjectUnloaded);

        return () => {
            editorManager.off('objectCreated', handleObjectCreated);
            editorManager.off('objectDeleted', handleObjectDeleted);
            editorManager.off('selectionChanged', handleSelectionChanged);
            sceneLoader.off('objectLoaded', handleObjectLoaded);
            sceneLoader.off('objectUnloaded', handleObjectUnloaded);
        };
    }, [editorManager]);

    // Focus input when editing starts
    useEffect(() => {
        if (editingId && inputRef.current) {
            inputRef.current.focus();
            inputRef.current.select();
        }
    }, [editingId]);

    const buildTree = () => {
        if (!editorManager?.scene) return;

        const buildNode = (object: any, depth: number = 0): TreeNode => {
            const children: TreeNode[] = [];

            // Process children
            if (object.children && object.children.length > 0) {
                object.children.forEach((child: any) => {
                    // Filter out helpers
                    if (!child.name.includes('Grid') && !child.name.includes('Axes')) {
                        children.push(buildNode(child, depth + 1));
                    }
                });
            }

            return {
                object,
                children,
                expanded: depth < 2, // Auto-expand first 2 levels
                visible: object.visible !== false,
                locked: object.userData?.locked || false,
                depth
            };
        };

        // Build tree from scene root (excluding helpers)
        const rootNodes: TreeNode[] = [];
        editorManager.scene.children.forEach((child: any) => {
            if (!child.name.includes('Grid') && !child.name.includes('Axes')) {
                rootNodes.push(buildNode(child, 0));
            }
        });

        setTreeNodes(rootNodes);
    };

    const handleObjectClick = (object: any, event: React.MouseEvent) => {
        event.stopPropagation();

        if (event.ctrlKey || event.metaKey) {
            // Multi-selection
            const selectionManager = SelectionManager.getInstance();
            selectionManager.addToSelection(object);
        } else {
            // Single selection
            const selectionManager = SelectionManager.getInstance();
            selectionManager.select(object);
            editorManager.selectObject(object);
        }

        if (onObjectSelected) {
            onObjectSelected(object);
        }
    };

    const handleDoubleClick = (object: any, event: React.MouseEvent) => {
        event.stopPropagation();
        startEditing(object);
    };

    const startEditing = (object: any) => {
        setEditingId(object.uuid);
        setEditingName(object.name || '');
    };

    const handleNameChange = (object: any) => {
        if (editingName.trim()) {
            object.name = editingName.trim();
        }
        setEditingId(null);
        setEditingName('');
    };

    const handleKeyDown = (object: any, e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleNameChange(object);
        } else if (e.key === 'Escape') {
            setEditingId(null);
            setEditingName('');
        }
    };

    const toggleVisibility = (object: any, event: React.MouseEvent) => {
        event.stopPropagation();
        object.visible = !object.visible;
        buildTree();
    };

    const toggleLock = (object: any, event: React.MouseEvent) => {
        event.stopPropagation();
        object.userData = object.userData || {};
        object.userData.locked = !object.userData.locked;
        buildTree();
    };

    const toggleExpand = (nodeIndex: number, path: number[] = []) => {
        const toggleNode = (nodes: TreeNode[], index: number, currentPath: number[]): TreeNode[] => {
            const newNodes = [...nodes];
            const originalNode = newNodes[index];

            if (!originalNode) return newNodes;

            const node: TreeNode = {
                object: originalNode.object,
                children: originalNode.children || [],
                expanded: originalNode.expanded,
                visible: originalNode.visible,
                locked: originalNode.locked,
                depth: originalNode.depth
            };

            if (currentPath.length === 0) {
                // Toggle this node
                node.expanded = !node.expanded;
            } else {
                // Traverse to children
                const [nextPath, ...remainingPath] = currentPath;
                if (node.children && nextPath !== undefined) {
                    node.children = toggleNode(node.children, nextPath, remainingPath);
                }
            }

            newNodes[index] = node;
            return newNodes;
        };

        setTreeNodes(toggleNode(treeNodes, nodeIndex, path));
    };

    const getObjectIcon = (object: any): string => {
        if (object.type && object.type.includes('Mesh')) {
            return '▢'; // Mesh icon
        } else if (object.type && object.type.includes('Light')) {
            return '💡'; // Light icon
        } else if (object.type && object.type.includes('Camera')) {
            return '📷'; // Camera icon
        } else if (object.type && object.type.includes('Group')) {
            return '📁'; // Group icon
        } else if (object.constructor?.name === 'SplatMesh' || object.type?.includes('Splat')) {
            return '🌐'; // Splat/Gaussian Splat icon
        }
        return '○'; // Default icon
    };

    const renderTreeNode = (node: TreeNode, path: number[] = []): React.ReactNode => {
        const isSelected = selectedObject === node.object;
        const isEditing = editingId === node.object.uuid;
        const hasChildren = node.children.length > 0;

        return (
            <div key={node.object.uuid} className="tree-node">
                <div
                    className={`hierarchy-item ${isSelected ? 'selected' : ''} ${isEditing ? 'editing' : ''}`}
                    style={{ paddingLeft: `${node.depth * 16 + 8}px` }}
                    onClick={(e) => handleObjectClick(node.object, e)}
                    onDoubleClick={(e) => handleDoubleClick(node.object, e)}
                >
                    {/* Expand/Collapse */}
                    {hasChildren && (
                        <span
                            className="expand-icon"
                            onClick={(e) => {
                                e.stopPropagation();
                                const idx = path[path.length - 1];
                                if (idx !== undefined) {
                                    toggleExpand(idx, path.slice(0, -1));
                                }
                            }}
                        >
                            {node.expanded ? '▼' : '▶'}
                        </span>
                    )}

                    {!hasChildren && <span className="expand-placeholder"></span>}

                    {/* Object Type Icon */}
                    <span className="object-icon">{getObjectIcon(node.object)}</span>

                    {/* Object Name */}
                    {isEditing ? (
                        <input
                            ref={inputRef}
                            type="text"
                            value={editingName}
                            onChange={(e) => setEditingName(e.target.value)}
                            onBlur={() => handleNameChange(node.object)}
                            onKeyDown={(e) => handleKeyDown(node.object, e)}
                            className="hierarchy-name-input"
                            onClick={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <span className="object-name">
                            {node.object.name || 'Unnamed Object'}
                        </span>
                    )}

                    {/* Actions */}
                    <div className="hierarchy-actions">
                        {/* Visibility Toggle */}
                        <span
                            className={`action-icon ${node.visible ? 'visible' : 'hidden'}`}
                            onClick={(e) => toggleVisibility(node.object, e)}
                            title={node.visible ? 'Hide' : 'Show'}
                        >
                            {node.visible ? '👁️' : '👁️‍🗨️'}
                        </span>

                        {/* Lock Toggle */}
                        <span
                            className={`action-icon ${node.locked ? 'locked' : 'unlocked'}`}
                            onClick={(e) => toggleLock(node.object, e)}
                            title={node.locked ? 'Unlock' : 'Lock'}
                        >
                            {node.locked ? '🔒' : '🔓'}
                        </span>
                    </div>
                </div>

                {/* Children */}
                {hasChildren && node.expanded && (
                    <div className="tree-children">
                        {node.children.map((child, index) =>
                            renderTreeNode(child, [...path, index])
                        )}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="hierarchy-container">
            <div className="hierarchy-header">
                <h3>Hierarchy</h3>
                <div className="hierarchy-toolbar">
                    <button
                        className="toolbar-button"
                        onClick={buildTree}
                        title="Refresh Hierarchy"
                    >
                        🔄
                    </button>
                    <button
                        className="toolbar-button"
                        onClick={() => {
                            const selectionManager = SelectionManager.getInstance();
                            selectionManager.clearSelection();
                        }}
                        title="Clear Selection"
                    >
                        ❌
                    </button>
                </div>
            </div>

            <div className="hierarchy-content">
                {treeNodes.length === 0 ? (
                    <div className="hierarchy-empty">
                        <p>No objects in scene</p>
                        <p className="hint">Create objects using the viewport toolbar</p>
                    </div>
                ) : (
                    <div className="hierarchy-tree">
                        {treeNodes.map((node, index) => renderTreeNode(node, [index]))}
                    </div>
                )}
            </div>

            <div className="hierarchy-footer">
                <small className="hint">
                    Double-click to rename • Ctrl+Click for multi-select
                </small>
            </div>
        </div>
    );
};

export default Hierarchy;
