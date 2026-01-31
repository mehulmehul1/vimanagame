/**
 * SceneNode.tsx - Custom ReactFlow Node for Scene Flow Navigator
 * =============================================================================
 *
 * Custom node component that displays game state information in the flow graph.
 * Each node shows:
 * - State name (formatted)
 * - State value (numeric)
 * - Category color coding
 * - Criteria badge (if entry criteria exist)
 * - Current state highlight
 * - Zone assignment
 * - Content indicators (video, dialog, music)
 */

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import type { StoryState, StateContent } from '../types/story';
import './SceneNode.css';

export interface SceneNodeData {
  label: string;
  stateName: string;
  stateValue: number;
  category: string;
  color: string;
  description?: string;
  hasCriteria?: boolean;
  isCurrent?: boolean;
  onJumpToState?: (stateName: string) => void;

  // Enhanced story data
  zone?: string | null;
  content?: StateContent;
  transitions?: any[];
  act?: number;
}

/**
 * Custom SceneNode component for ReactFlow
 * Displays game state as a colored node in the flow diagram
 */
const SceneNode: React.FC<NodeProps<SceneNodeData>> = ({ data, selected }) => {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data.onJumpToState) {
      data.onJumpToState(data.stateName);
    }
  };

  /**
   * Check if state has content attachments
   */
  const hasContent = () => {
    if (!data.content) return false;
    return !!(data.content.video || data.content.dialog || data.content.music);
  };

  /**
   * Get content indicators
   */
  const getContentIndicators = () => {
    const indicators: JSX.Element[] = [];
    if (!data.content) return indicators;

    if (data.content.video) {
      indicators.push(<span key="video" className="content-indicator video" title={`Video: ${data.content.video}`}>📹</span>);
    }
    if (data.content.dialog) {
      indicators.push(<span key="dialog" className="content-indicator dialog" title={`Dialog: ${data.content.dialog}`}>💬</span>);
    }
    if (data.content.music) {
      indicators.push(<span key="music" className="content-indicator music" title={`Music: ${data.content.music}`}>🎵</span>);
    }

    return indicators;
  };

  return (
    <div
      className={`scene-node ${selected ? 'selected' : ''} ${data.isCurrent ? 'current' : ''}`}
      style={
        {
          '--node-color': data.color,
        } as React.CSSProperties
      }
      onClick={handleClick}
      title={data.description || `Jump to ${data.label}`}
    >
      {/* Input handle (left side) */}
      <Handle type="target" position={Position.Left} className="node-handle target" />

      {/* Node content */}
      <div className="scene-node-content">
        {/* State header */}
        <div className="scene-node-header">
          <span className="scene-node-value">{data.stateValue}</span>
          {data.hasCriteria && (
            <span className="scene-node-badge" title="Has entry criteria">
              ◈
            </span>
          )}
        </div>

        {/* State label */}
        <div className="scene-node-label">{data.label}</div>

        {/* Zone indicator */}
        {data.zone && (
          <div className="scene-node-zone" title={`Zone: ${data.zone}`}>
            📍 {data.zone}
          </div>
        )}

        {/* Category indicator */}
        <div className="scene-node-category" title={data.category}>
          {data.category}
        </div>

        {/* Content indicators */}
        {hasContent() && (
          <div className="scene-node-content-indicators">
            {getContentIndicators()}
          </div>
        )}

        {/* Transition count */}
        {data.transitions && data.transitions.length > 0 && (
          <div className="scene-node-transitions" title={`${data.transitions.length} outgoing transition(s)`}>
            ↗ {data.transitions.length}
          </div>
        )}

        {/* Jump hint */}
        <div className="scene-node-hint">Double-click to jump</div>
      </div>

      {/* Output handle (right side) */}
      <Handle type="source" position={Position.Right} className="node-handle source" />
    </div>
  );
};

export default memo(SceneNode);
