/**
 * SceneFlowNavigator.tsx - Scene Flow Navigator Panel
 * =============================================================================
 *
 * Visual story editing panel using ReactFlow to display the game's narrative
 * flow as an interactive node graph.
 *
 * Features:
 * - Visualizes all game states as nodes in a flow graph
 * - Color coding by category (Intro=Blue, Exterior=Green, etc.)
 * - Double-click to jump to any scene
 * - Highlights current scene
 * - Shows entry criteria on nodes
 * - Keyboard navigation (arrow keys + Enter)
 * - Auto-layout left-to-right by state value
 * - State Inspector panel for editing
 * - Story data management with undo/redo
 */

import React, { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ConnectionMode,
  useNodesState,
  useEdgesState,
  addEdge,
  Panel,
} from 'reactflow';
import type { Node, Edge, NodeTypes } from 'reactflow';
import 'reactflow/dist/style.css';
import SceneNode, { type SceneNodeData } from '../../components/SceneNode';
import StateInspector from '../../components/StateInspector';
import { getStoryDataManager } from '../../managers/StoryStateManager';
import { loadStoryData, saveStoryData, generateBackupFilename, exportToJSON } from '../../utils/storyDataIO';
import {
  GAME_STATES,
  STATE_CATEGORIES,
  formatStateName,
} from '../../data/gameStateData';
import type { StoryState, ZoneInfo, PlayerPosition, TriggerType } from '../../types/story';
import './SceneFlowNavigator.css';

// Node types registration
const nodeTypes: NodeTypes = {
  sceneNode: SceneNode,
};

// Available zones (will be integrated with SceneManager later)
const AVAILABLE_ZONES: ZoneInfo[] = [
  { id: 'plaza', name: 'Plaza', color: '#22c55e' },
  { id: 'street', name: 'Street', color: '#f59e0b' },
  { id: 'office_exterior', name: 'Office Exterior', color: '#64748b' },
  { id: 'office_interior', name: 'Office Interior', color: '#8b5cf6' },
];

/**
 * SceneFlowNavigator Panel Component
 */
const SceneFlowNavigator: React.FC<{
  currentStateValue?: number;
  onJumpToState?: (stateName: string) => void;
}> = ({ currentStateValue, onJumpToState }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<SceneNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedState, setSelectedState] = useState<StoryState | null>(null);
  const [showInspector, setShowInspector] = useState(true);
  const [storyDataLoaded, setStoryDataLoaded] = useState(false);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Get singleton StoryDataManager instance
  const storyManager = getStoryDataManager();

  /**
   * Jump to a specific game state
   */
  const handleJumpToState = useCallback(
    (stateName: string) => {
      console.log(`SceneFlowNavigator: Jumping to state ${stateName}`);

      // Update URL with gameState parameter
      const url = new URL(window.location.href);
      url.searchParams.set('gameState', stateName);

      // Reload page with new state
      window.location.href = url.toString();

      // Also call the provided callback
      if (onJumpToState) {
        onJumpToState(stateName);
      }
    },
    [onJumpToState]
  );

  /**
   * Update state from inspector
   */
  const handleUpdateState = useCallback((updates: Partial<StoryState>) => {
    if (selectedNode) {
      storyManager.updateState(selectedNode, updates);
      // Refresh selected state
      const updated = storyManager.getState(selectedNode);
      setSelectedState(updated);
    }
  }, [selectedNode, storyManager]);

  /**
   * Delete state from inspector
   */
  const handleDeleteState = useCallback(() => {
    if (selectedNode && confirm(`Are you sure you want to delete state "${selectedNode}"?`)) {
      storyManager.deleteState(selectedNode);
      setSelectedNode(null);
      setSelectedState(null);
    }
  }, [selectedNode, storyManager]);

  /**
   * Duplicate state from inspector
   */
  const handleDuplicateState = useCallback(() => {
    if (selectedState) {
      const newId = `${selectedState.id}_COPY`;
      const newLabel = `${selectedState.label} (Copy)`;
      storyManager.duplicateState(selectedState.id, newId, newLabel);
    }
  }, [selectedState, storyManager]);

  /**
   * Test state from inspector
   */
  const handleTestState = useCallback(() => {
    if (selectedNode) {
      handleJumpToState(selectedNode);
    }
  }, [selectedNode, handleJumpToState]);

  /**
   * Jump to zone in 3D viewport
   */
  const handleJumpToZone = useCallback((zoneId: string) => {
    console.log(`SceneFlowNavigator: Jump to zone ${zoneId}`);
    // TODO: Integrate with 3D viewport camera
  }, []);

  /**
   * Capture player position from viewport
   */
  const handleCapturePosition = useCallback((): PlayerPosition | null => {
    // TODO: Integrate with 3D viewport to get actual player position
    // For now, return a default position
    return {
      x: 0,
      y: 1.6,
      z: 0,
      rotation: { x: 0, y: 0, z: 0 },
    };
  }, []);

  /**
   * Handle undo
   */
  const handleUndo = useCallback(() => {
    storyManager.undo();
  }, [storyManager]);

  /**
   * Handle redo
   */
  const handleRedo = useCallback(() => {
    storyManager.redo();
  }, [storyManager]);

  /**
   * Add new state
   */
  const handleAddState = useCallback(() => {
    const newId = `NEW_STATE_${Date.now()}`;
    const newValue = storyManager.getNextStateValue();
    const newState: StoryState = {
      id: newId,
      value: newValue,
      label: `New State ${newValue}`,
      act: 1,
      category: 'Intro & Title',
      color: '#64748b',
      zone: null,
      content: {},
      transitions: [],
      description: 'New story state',
    };
    storyManager.addState(newState);
    setSelectedNode(newId);
    setSelectedState(newState);
  }, [storyManager]);

  /**
   * Export story data to JSON file
   */
  const handleExport = useCallback(async () => {
    try {
      const storyData = storyManager.getData();
      await saveStoryData(storyData, generateBackupFilename('storyData'));
      console.log('Story data exported successfully');
    } catch (error) {
      console.error('Failed to export story data:', error);
      alert('Failed to export story data: ' + (error instanceof Error ? error.message : String(error)));
    }
  }, [storyManager]);

  /**
   * Import story data from JSON file
   */
  const handleImport = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        const content = await file.text();
        const data = JSON.parse(content);

        // Validate and update story manager
        for (const [stateId, state] of Object.entries(data.states)) {
          storyManager.addState(state as StoryState);
        }

        // Refresh nodes and edges
        setStoryDataLoaded(false);
        setTimeout(() => setStoryDataLoaded(true), 50);

        console.log('Story data imported successfully');
      } catch (error) {
        console.error('Failed to import story data:', error);
        alert('Failed to import story data: ' + (error instanceof Error ? error.message : String(error)));
      }
    };
    input.click();
  }, [storyManager]);

  /**
   * Generate nodes from story data
   * Creates a node for each story state with position based on act grouping
   */
  const generateNodes = useCallback(() => {
    const generatedNodes = [];
    const storyData = storyManager.getData();
    const categoryKeys = Object.keys(STATE_CATEGORIES);

    for (const [stateId, state] of Object.entries(storyData.states)) {
      // Calculate position based on act and value (left-to-right flow)
      const categoryIndex = categoryKeys.indexOf(state.category);
      const valueIndex = state.value + 1; // Offset for negative values

      generatedNodes.push({
        id: stateId,
        type: 'sceneNode',
        position: {
          x: valueIndex * 200, // Horizontal spacing by state value
          y: (state.act - 1) * 150 + categoryIndex * 80, // Vertical spacing by act and category
        },
        data: {
          label: state.label,
          stateName: stateId,
          stateValue: state.value,
          category: state.category,
          color: state.color,
          description: state.description,
          hasCriteria: !!state.criteria,
          isCurrent: state.value === currentStateValue,
          onJumpToState: handleJumpToState,
          zone: state.zone,
          content: state.content,
          transitions: state.transitions,
          act: state.act,
        },
      });
    }

    return generatedNodes;
  }, [currentStateValue, handleJumpToState, storyManager]);

  /**
   * Generate edges from story data transitions
   * Creates connections based on the transitions defined in each state
   */
  const generateEdges = useCallback(() => {
    const generatedEdges = [];
    const storyData = storyManager.getData();

    // Color coding by trigger type
    const triggerColors: Record<TriggerType, string> = {
      onComplete: '#94a3b8', // gray
      onChoice: '#3b82f6', // blue
      onTimeout: '#f59e0b', // amber
      onProximity: '#22c55e', // green
      onInteract: '#a855f7', // purple
      onState: '#ef4444', // red
      custom: '#ec4899', // pink
    };

    // Generate edges from each state's transitions
    for (const [fromStateId, state] of Object.entries(storyData.states)) {
      for (const transition of state.transitions) {
        const edgeColor = triggerColors[transition.trigger] || '#94a3b8';

        generatedEdges.push({
          id: transition.id,
          source: fromStateId,
          target: transition.to,
          type: 'smoothstep',
          animated: transition.trigger === 'onComplete', // Animate primary transitions
          style: {
            stroke: edgeColor,
            strokeWidth: 2,
          },
          label: transition.label,
          labelStyle: {
            fontSize: 10,
            fill: '#94a3b8',
          },
          data: {
            trigger: transition.trigger,
          },
        });
      }
    }

    return generatedEdges;
  }, [storyManager]);

  /**
   * Handle new connections (for manual edge creation)
   */
  const onConnect = useCallback(
    (params: any) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  /**
   * Handle node selection
   */
  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
    // Load state from story manager
    const state = storyManager.getState(node.id);
    setSelectedState(state || null);
  }, [storyManager]);

  /**
   * Handle double-click to jump to state
   */
  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node<SceneNodeData>) => {
      handleJumpToState(node.data.stateName);
    },
    [handleJumpToState]
  );

  /**
   * Handle keyboard navigation
   */
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Only handle if not in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (selectedNode) {
        const storyData = storyManager.getData();
        const sortedStates = Object.values(storyData.states)
          .filter(s => s.id !== 'LOADING' && s.id !== 'GAME_OVER')
          .sort((a, b) => a.value - b.value);
        const currentIndex = sortedStates.findIndex((s) => s.id === selectedNode);

        switch (e.key) {
          case 'ArrowRight':
          case 'ArrowDown':
            e.preventDefault();
            if (currentIndex < sortedStates.length - 1) {
              const nextNode = sortedStates[currentIndex + 1];
              if (nextNode) {
                setSelectedNode(nextNode.id);
                const categoryKeys = Object.keys(STATE_CATEGORIES);
                reactFlowInstance?.setCenter(
                  nextNode.value * 200,
                  (nextNode.act - 1) * 150 + categoryKeys.indexOf(nextNode.category) * 80,
                  { zoom: 1, duration: 300 }
                );
              }
            }
            break;
          case 'ArrowLeft':
          case 'ArrowUp':
            e.preventDefault();
            if (currentIndex > 0) {
              const prevNode = sortedStates[currentIndex - 1];
              if (prevNode) {
                setSelectedNode(prevNode.id);
                const categoryKeys = Object.keys(STATE_CATEGORIES);
                reactFlowInstance?.setCenter(
                  prevNode.value * 200,
                  (prevNode.act - 1) * 150 + categoryKeys.indexOf(prevNode.category) * 80,
                  { zoom: 1, duration: 300 }
                );
              }
            }
            break;
          case 'Enter':
            e.preventDefault();
            handleJumpToState(selectedNode);
            break;
        }
      }
    },
    [selectedNode, reactFlowInstance, handleJumpToState, storyManager]
  );

  /**
   * Load story data on mount
   */
  useEffect(() => {
    const loadStoryDataOnMount = async () => {
      try {
        await storyManager.load('editor/data/storyData.json');
        setStoryDataLoaded(true);
      } catch (error) {
        console.error('Failed to load storyData.json, using default data:', error);
        // If loading fails, use the default data in storyManager
        setStoryDataLoaded(true);
      }
    };
    loadStoryDataOnMount();
  }, [storyManager]);

  /**
   * Refresh nodes and edges when story data changes
   */
  useEffect(() => {
    if (!storyDataLoaded) return;

    const initialNodes = generateNodes();
    const initialEdges = generateEdges();

    setNodes(initialNodes);
    setEdges(initialEdges);

    // Fit view to show all nodes
    if (reactFlowInstance) {
      setTimeout(() => {
        reactFlowInstance.fitView({ padding: 0.2, duration: 300 });
      }, 100);
    }
  }, [storyDataLoaded, generateNodes, generateEdges, setNodes, setEdges, reactFlowInstance]);

  /**
   * Update current state highlight
   */
  useEffect(() => {
    setNodes((nodes) =>
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          isCurrent: (node.data as SceneNodeData).stateValue === currentStateValue,
        },
      }))
    );
  }, [currentStateValue, setNodes]);

  /**
   * Register keyboard listeners
   */
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  /**
   * Fit view on mount
   */
  const onInit = useCallback((instance: any) => {
    setReactFlowInstance(instance);
    setTimeout(() => {
      instance.fitView({ padding: 0.2, duration: 300 });
    }, 100);
  }, []);

  /**
   * Get current state name for display
   */
  const getCurrentStateName = useMemo(() => {
    if (currentStateValue === null || currentStateValue === undefined) {
      return 'None';
    }
    const storyData = storyManager.getData();
    const state = Object.values(storyData.states).find(s => s.value === currentStateValue);
    return state ? state.label : 'Unknown';
  }, [currentStateValue, storyManager]);

  return (
    <div className="scene-flow-navigator-container">
      <div className="scene-flow-navigator" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onNodeDoubleClick={onNodeDoubleClick}
          onInit={onInit}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          fitView
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
          minZoom={0.2}
          maxZoom={2}
          className="scene-flow-canvas"
        >
          <Background color="#374151" gap={16} />
          <Controls className="scene-flow-controls" />

          <MiniMap
            nodeColor={(node) => (node.data as SceneNodeData).color}
            className="scene-flow-minimap"
          />

          {/* Info Panel */}
          <Panel position="top-left" className="scene-flow-info">
            <div className="scene-flow-header">
              <h3>Scene Flow Navigator</h3>
              <div className="scene-flow-current">
                <span className="current-label">Current:</span>
                <span className="current-value">{getCurrentStateName}</span>
              </div>
            </div>
          </Panel>

          {/* Toolbar */}
          <Panel position="top-center" className="scene-flow-toolbar">
            <div className="toolbar-actions">
              <button
                className="toolbar-btn"
                onClick={handleAddState}
                title="Add new state"
              >
                ➕ Add State
              </button>
              <button
                className="toolbar-btn"
                onClick={handleUndo}
                disabled={!storyManager.canUndo()}
                title="Undo (Ctrl+Z)"
              >
                ↶ Undo
              </button>
              <button
                className="toolbar-btn"
                onClick={handleRedo}
                disabled={!storyManager.canRedo()}
                title="Redo (Ctrl+Y)"
              >
                ↷ Redo
              </button>
              <button
                className="toolbar-btn"
                onClick={handleExport}
                title="Export story data to JSON"
              >
                💾 Export
              </button>
              <button
                className="toolbar-btn"
                onClick={handleImport}
                title="Import story data from JSON"
              >
                📂 Import
              </button>
              <button
                className="toolbar-btn"
                onClick={() => setShowInspector(!showInspector)}
                title={showInspector ? 'Hide Inspector' : 'Show Inspector'}
              >
                {showInspector ? '📋 Hide' : '📋 Inspector'}
              </button>
            </div>
          </Panel>

          {/* Legend Panel */}
          <Panel position="bottom-right" className="scene-flow-legend">
            <div className="legend-title">Categories</div>
            <div className="legend-items">
              {Object.entries(STATE_CATEGORIES).map(([category, data]) => (
                <div key={category} className="legend-item">
                  <span
                    className="legend-color"
                    style={{ backgroundColor: data.color }}
                  />
                  <span className="legend-label">{category}</span>
                </div>
              ))}
            </div>
          </Panel>

          {/* Help Panel */}
          <Panel position="top-right" className="scene-flow-help">
            <div className="help-item">
              <kbd>Double-click</kbd> node to jump
            </div>
            <div className="help-item">
              <kbd>Arrows</kbd> to navigate
            </div>
            <div className="help-item">
              <kbd>Enter</kbd> to confirm
            </div>
          </Panel>
        </ReactFlow>
      </div>

      {/* State Inspector Panel */}
      {showInspector && (
        <div className="scene-flow-inspector-panel">
          <StateInspector
            state={selectedState}
            availableZones={AVAILABLE_ZONES}
            onUpdateState={handleUpdateState}
            onDeleteState={handleDeleteState}
            onDuplicateState={handleDuplicateState}
            onTestState={handleTestState}
            onJumpToZone={handleJumpToZone}
            onCapturePosition={handleCapturePosition}
          />
        </div>
      )}
    </div>
  );
};

export default SceneFlowNavigator;
