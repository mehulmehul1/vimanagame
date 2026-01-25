import React, { useCallback, useMemo, useState } from 'react';
import ReactFlow, {
    addEdge,
    useNodesState,
    useEdgesState,
    Controls,
    Background,
    MiniMap,
    MarkerType,
    applyNodeChanges,
    applyEdgeChanges,
} from 'reactflow';
import type { Node, Edge, NodeTypes, NodeChange, EdgeChange } from 'reactflow';
import 'reactflow/dist/style.css';
import './NodeGraph.css';

// Connection type for addEdge callback
interface Connection {
    source: string | null;
    target: string | null;
    sourceHandle?: string | null;
    targetHandle?: string | null;
}

/**
 * Node data types
 */
interface NodeData {
    label: string;
    type: 'event' | 'action' | 'condition' | 'variable';
    category: string;
    code?: string;
    properties?: Record<string, any>;
}

/**
 * Custom node component
 */
const CustomNode: React.FC<{ data: NodeData; id: string }> = ({ data, id }) => {
    const getNodeColor = (type: string) => {
        switch (type) {
            case 'event': return '#4a9eff';
            case 'action': return '#51cf66';
            case 'condition': return '#ffd43b';
            case 'variable': return '#ff922b';
            default: return '#888';
        }
    };

    const getNodeIcon = (type: string) => {
        switch (type) {
            case 'event': return '⚡';
            case 'action': return '▶';
            case 'condition': return '?';
            case 'variable': return '★';
            default: return '○';
        }
    };

    return (
        <div
            className="custom-node"
            style={{ '--node-color': getNodeColor(data.type) } as React.CSSProperties}
        >
            <div className="node-header">
                <span className="node-icon">{getNodeIcon(data.type)}</span>
                <span className="node-label">{data.label}</span>
            </div>
            <div className="node-category">{data.category}</div>
            {data.properties && (
                <div className="node-properties">
                    {Object.entries(data.properties).slice(0, 3).map(([key, value]) => (
                        <div key={key} className="property-item">
                            <span className="property-key">{key}:</span>
                            <span className="property-value">{String(value).slice(0, 15)}</span>
                        </div>
                    ))}
                </div>
            )}
            <Handle type="target" position="left" />
            <Handle type="source" position="right" />
        </div>
    );
};

// Handle component for ReactFlow
const Handle = ({ type, position }: { type: 'source' | 'target'; position: 'left' | 'right' }) => {
    return (
        <div
            className={`react-flow__handle react-flow__handle-${type} react-flow__handle-${position}`}
            data-handleid={`${type}-${position}`}
        />
    );
};

const nodeTypes: NodeTypes = {
    custom: CustomNode as any,
};

/**
 * Node templates
 */
const NODE_TEMPLATES: Array<{
    type: NodeData['type'];
    category: string;
    label: string;
    properties?: Record<string, any>;
    code?: string;
}> = [
    // Event nodes
    { type: 'event', category: 'Events', label: 'On State Changed', properties: { stateName: '' } },
    { type: 'event', category: 'Events', label: 'On Enter', properties: { trigger: 'enter' } },
    { type: 'event', category: 'Events', label: 'On Exit', properties: { trigger: 'exit' } },
    { type: 'event', category: 'Events', label: 'On Click', properties: { targetId: '' } },
    { type: 'event', category: 'Events', label: 'On Collision', properties: { objectId: '' } },

    // Action nodes
    { type: 'action', category: 'Actions', label: 'Set State', properties: { state: '', value: '' } },
    { type: 'action', category: 'Actions', label: 'Show Dialog', properties: { dialogId: '' } },
    { type: 'action', category: 'Actions', label: 'Play Sound', properties: { soundId: '', volume: 1 } },
    { type: 'action', category: 'Actions', label: 'Load Scene', properties: { sceneId: '' } },
    { type: 'action', category: 'Actions', label: 'Wait', properties: { duration: 1 } },
    { type: 'action', category: 'Actions', label: 'Debug Log', properties: { message: '' } },

    // Condition nodes
    { type: 'condition', category: 'Conditions', label: 'Criteria Check', properties: { criteria: '' } },
    { type: 'condition', category: 'Conditions', label: 'Has Item', properties: { itemId: '' } },
    { type: 'condition', category: 'Conditions', label: 'State Equals', properties: { state: '', value: '' } },
    { type: 'condition', category: 'Conditions', label: 'Random Chance', properties: { chance: 50 } },

    // Variable nodes
    { type: 'variable', category: 'Variables', label: 'Get Variable', properties: { varName: '' } },
    { type: 'variable', category: 'Variables', label: 'Set Variable', properties: { varName: '', value: '' } },
    { type: 'variable', category: 'Variables', label: 'Increment', properties: { varName: '', amount: 1 } },
];

/**
 * NodeGraph Panel - Visual scripting with ReactFlow
 *
 * Features:
 * - Event nodes: state:changed, onEnter, onExit, onClick
 * - Action nodes: setState, showDialog, playSound, loadScene
 * - Condition nodes: criteria builder (AND/OR logic)
 * - Code generation to JavaScript
 */
interface NodeGraphProps {
    editorManager?: any;
    onGenerateCode?: (code: string) => void;
}

const NodeGraph: React.FC<NodeGraphProps> = ({ editorManager, onGenerateCode }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState<NodeData>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [selectedNode, setSelectedNode] = useState<Node<NodeData> | null>(null);

    /**
     * Add a new node from template
     */
    const addNode = useCallback((template: typeof NODE_TEMPLATES[number], position?: { x: number; y: number }) => {
        const newNode: Node<NodeData> = {
            id: `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            type: 'custom',
            position: position || { x: Math.random() * 400 + 100, y: Math.random() * 300 + 100 },
            data: {
                label: template.label,
                type: template.type,
                category: template.category,
                properties: { ...template.properties },
            },
        };

        setNodes((nds) => [...nds, newNode]);
    }, [setNodes]);

    /**
     * Delete selected node
     */
    const deleteSelectedNode = useCallback(() => {
        if (selectedNode) {
            setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
            setSelectedNode(null);
        }
    }, [selectedNode, setNodes]);

    /**
     * Update node data
     */
    const updateNodeData = useCallback((nodeId: string, updates: Partial<NodeData>) => {
        setNodes((nds) =>
            nds.map((node) =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, ...updates } }
                    : node
            )
        );
    }, [setNodes]);

    /**
     * Clear all nodes and edges
     */
    const clearGraph = useCallback(() => {
        setNodes([]);
        setEdges([]);
    }, [setNodes, setEdges]);

    /**
     * Generate JavaScript code from the node graph
     */
    const generateCode = useCallback(() => {
        let code = '// Generated scene logic\n// Auto-generated by Shadow Web Editor Node Graph\n\n';

        // Find all event nodes (starting points)
        const eventNodes = nodes.filter((n) => n.data.type === 'event');

        eventNodes.forEach((eventNode) => {
            const { label, properties } = eventNode.data;

            switch (label) {
                case 'On State Changed':
                    code += `game.on('stateChanged', (newState) => {\n`;
                    code += `    if (newState === '${properties?.stateName || ''}') {\n`;
                    code += generateNodeLogic(eventNode);
                    code += `    }\n`;
                    code += `});\n\n`;
                    break;
                case 'On Enter':
                    code += `scene.on('enter', () => {\n`;
                    code += generateNodeLogic(eventNode);
                    code += `});\n\n`;
                    break;
                case 'On Exit':
                    code += `scene.on('exit', () => {\n`;
                    code += generateNodeLogic(eventNode);
                    code += `});\n\n`;
                    break;
                case 'On Click':
                    code += `${properties?.targetId || 'object'}.on('click', () => {\n`;
                    code += generateNodeLogic(eventNode);
                    code += `});\n\n`;
                    break;
                default:
                    code += `// Event: ${label}\n`;
                    code += generateNodeLogic(eventNode);
            }
        });

        return code;
    }, [nodes, edges]);

    /**
     * Generate logic for a node and its connected nodes
     */
    const generateNodeLogic = (node: Node<NodeData>, visited = new Set<string>()): string => {
        if (visited.has(node.id)) return '';
        visited.add(node.id);

        let logic = '';

        switch (node.data.type) {
            case 'action':
                logic += generateActionCode(node.data);
                break;
            case 'condition':
                logic += generateConditionCode(node.data);
                break;
            case 'variable':
                logic += generateVariableCode(node.data);
                break;
        }

        // Find connected nodes
        const connectedEdges = edges.filter((e) => e.source === node.id);
        connectedEdges.forEach((edge) => {
            const targetNode = nodes.find((n) => n.id === edge.target);
            if (targetNode) {
                logic += generateNodeLogic(targetNode, visited);
            }
        });

        return logic;
    };

    /**
     * Generate action code
     */
    const generateActionCode = (data: NodeData): string => {
        const { label, properties } = data;

        switch (label) {
            case 'Set State':
                return `    game.setState('${properties?.state || ''}');\n`;
            case 'Show Dialog':
                return `    ui.showDialog('${properties?.dialogId || ''}');\n`;
            case 'Play Sound':
                return `    audio.play('${properties?.soundId || ''}', ${properties?.volume || 1});\n`;
            case 'Load Scene':
                return `    scene.load('${properties?.sceneId || ''}');\n`;
            case 'Wait':
                return `    await util.wait(${properties?.duration || 1});\n`;
            case 'Debug Log':
                return `    console.log('${properties?.message || ''}');\n`;
            default:
                return `    // Unknown action: ${label}\n`;
        }
    };

    /**
     * Generate condition code
     */
    const generateConditionCode = (data: NodeData): string => {
        const { label, properties } = data;

        switch (label) {
            case 'Criteria Check':
                return `    if (criteria.check('${properties?.criteria || ''}')) {\n        // Continue\n    }\n`;
            case 'Has Item':
                return `    if (inventory.has('${properties?.itemId || ''}')) {\n        // Has item\n    }\n`;
            case 'State Equals':
                return `    if (game.getState() === '${properties?.state || ''}') {\n        // State matches\n    }\n`;
            case 'Random Chance':
                return `    if (Math.random() * 100 < ${properties?.chance || 50}) {\n        // Success\n    }\n`;
            default:
                return `    // Unknown condition: ${label}\n`;
        }
    };

    /**
     * Generate variable code
     */
    const generateVariableCode = (data: NodeData): string => {
        const { label, properties } = data;

        switch (label) {
            case 'Get Variable':
                return `    const value = variables.get('${properties?.varName || ''}');\n`;
            case 'Set Variable':
                return `    variables.set('${properties?.varName || ''}, ${properties?.value || ''});\n`;
            case 'Increment':
                return `    variables.increment('${properties?.varName || ''}', ${properties?.amount || 1});\n`;
            default:
                return `    // Unknown variable operation: ${label}\n`;
        }
    };

    /**
     * Handle connection
     */
    const onConnect = useCallback(
        (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
        [setEdges]
    );

    /**
     * Handle node selection
     */
    const onNodeClick = useCallback((_: React.MouseEvent, node: Node<NodeData>) => {
        setSelectedNode(node);
    }, []);

    /**
     * Handle drag over
     */
    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    /**
     * Handle drop from palette
     */
    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            const templateData = event.dataTransfer.getData('application/reactflow');
            if (!templateData) return;

            const template = JSON.parse(templateData);
            const reactFlowBounds = (event.target as Element).getBoundingClientRect();

            // Calculate position
            const position = {
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
            };

            addNode(template, position);
        },
        [addNode]
    );

    /**
     * Group templates by category
     */
    const groupedTemplates = useMemo(() => {
        const groups: Record<string, typeof NODE_TEMPLATES> = {};
        NODE_TEMPLATES.forEach((template) => {
            if (!groups[template.category]) {
                groups[template.category] = [] as any;
            }
            groups[template.category].push(template);
        });
        return groups;
    }, []);

    return (
        <div className="nodegraph-container">
            {/* Node Palette */}
            <div className="node-palette">
                <div className="palette-header">
                    <h4>Nodes</h4>
                    <button className="clear-btn" onClick={clearGraph} title="Clear graph">
                        🗑
                    </button>
                </div>
                <div className="palette-content">
                    {Object.entries(groupedTemplates).map(([category, templates]) => (
                        <div key={category} className="palette-category">
                            <div className="category-header">{category}</div>
                            {templates.map((template, index) => (
                                <div
                                    key={`${category}-${index}`}
                                    className="palette-node"
                                    draggable
                                    onDragStart={(e) => {
                                        e.dataTransfer.setData(
                                            'application/reactflow',
                                            JSON.stringify(template)
                                        );
                                        e.dataTransfer.effectAllowed = 'move';
                                    }}
                                    style={{ '--node-color':
                                        template.type === 'event' ? '#4a9eff' :
                                        template.type === 'action' ? '#51cf66' :
                                        template.type === 'condition' ? '#ffd43b' : '#ff922b'
                                    } as React.CSSProperties}
                                >
                                    <span className="palette-node-icon">
                                        {template.type === 'event' ? '⚡' :
                                         template.type === 'action' ? '▶' :
                                         template.type === 'condition' ? '?' : '★'}
                                    </span>
                                    <span className="palette-node-label">{template.label}</span>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            {/* Node Canvas */}
            <div className="node-canvas">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onDragOver={onDragOver}
                    onDrop={onDrop}
                    nodeTypes={nodeTypes}
                    fitView
                    defaultEdgeOptions={{
                        type: 'smoothstep',
                        animated: true,
                        markerEnd: { type: MarkerType.ArrowClosed },
                        style: { stroke: '#555' },
                    }}
                >
                    <Controls />
                    <MiniMap
                        nodeColor={(node) => {
                            const type = node.data.type;
                            return type === 'event' ? '#4a9eff' :
                                   type === 'action' ? '#51cf66' :
                                   type === 'condition' ? '#ffd43b' : '#ff922b';
                        }}
                        maskColor="rgba(0, 0, 0, 0.6)"
                    />
                    <Background color="#1a1a1a" gap={16} />
                </ReactFlow>
            </div>

            {/* Properties Panel */}
            <div className="node-properties">
                <div className="properties-header">
                    <h4>Properties</h4>
                    {selectedNode && (
                        <button className="delete-node-btn" onClick={deleteSelectedNode} title="Delete node">
                            🗑
                        </button>
                    )}
                </div>

                {selectedNode ? (
                    <div className="properties-content">
                        <div className="property-group">
                            <label>Label</label>
                            <input
                                type="text"
                                value={selectedNode.data.label}
                                onChange={(e) => updateNodeData(selectedNode.id, { label: e.target.value })}
                            />
                        </div>

                        <div className="property-group">
                            <label>Type</label>
                            <input
                                type="text"
                                value={selectedNode.data.type}
                                disabled
                            />
                        </div>

                        {selectedNode.data.properties && (
                            <div className="property-group">
                                <label>Properties</label>
                                {Object.entries(selectedNode.data.properties).map(([key, value]) => (
                                    <div key={key} className="property-row">
                                        <label>{key}</label>
                                        <input
                                            type="text"
                                            value={String(value)}
                                            onChange={(e) => {
                                                const newProps = { ...selectedNode.data.properties, [key]: e.target.value };
                                                updateNodeData(selectedNode.id, { properties: newProps });
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="property-group">
                            <label>Node ID</label>
                            <input
                                type="text"
                                value={selectedNode.id}
                                disabled
                            />
                        </div>
                    </div>
                ) : (
                    <div className="properties-empty">
                        <p>Select a node to edit its properties</p>
                    </div>
                )}

                {/* Generate Code Button */}
                <div className="code-generation">
                    <button
                        className="generate-code-btn"
                        onClick={() => {
                            const code = generateCode();
                            if (onGenerateCode) {
                                onGenerateCode(code);
                            } else {
                                console.log('Generated code:', code);
                                alert('Code generated! Check console for output.');
                            }
                        }}
                    >
                        Generate Code
                    </button>
                    <button
                        className="export-code-btn"
                        onClick={() => {
                            const code = generateCode();
                            const blob = new Blob([code], { type: 'text/javascript' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'sceneLogic.js';
                            a.click();
                            URL.revokeObjectURL(url);
                        }}
                    >
                        Export to File
                    </button>
                </div>
            </div>
        </div>
    );
};

export default NodeGraph;
