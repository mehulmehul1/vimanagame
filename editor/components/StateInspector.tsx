/**
 * StateInspector.tsx - State Inspector Panel for Scene Flow Navigator
 * =============================================================================
 *
 * Provides editing capabilities for selected story states including:
 * - Zone assignment
 * - Player position capture
 * - Content attachments (video, dialog, music)
 * - Entry criteria editing
 * - Transition management
 * - State testing and duplication
 */

import React, { useState, useEffect, useCallback } from 'react';
import type { StoryState, Transition, TriggerType, PlayerPosition, StateContent, ZoneInfo } from '../types/story';
import './StateInspector.css';

interface StateInspectorProps {
  state: StoryState | null;
  availableZones: ZoneInfo[];
  onUpdateState: (updates: Partial<StoryState>) => void;
  onDeleteState: () => void;
  onDuplicateState: () => void;
  onTestState: () => void;
  onJumpToZone?: (zoneId: string) => void;
  onCapturePosition?: () => PlayerPosition | null;
}

/**
 * StateInspector Component
 *
 * Displays and edits properties of a selected story state.
 */
const StateInspector: React.FC<StateInspectorProps> = ({
  state,
  availableZones,
  onUpdateState,
  onDeleteState,
  onDuplicateState,
  onTestState,
  onJumpToZone,
  onCapturePosition,
}) => {
  const [label, setLabel] = useState('');
  const [zone, setZone] = useState<string>('');
  const [description, setDescription] = useState('');
  const [notes, setNotes] = useState('');
  const [playerPos, setPlayerPos] = useState<PlayerPosition | null>(null);
  const [content, setContent] = useState<StateContent>({});
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['basic', 'zone', 'content']));

  /**
   * Initialize form when state changes
   */
  useEffect(() => {
    if (state) {
      setLabel(state.label || '');
      setZone(state.zone || '');
      setDescription(state.description || '');
      setNotes(state.notes || '');
      setPlayerPos(state.playerPosition || null);
      setContent(state.content || {});
    }
  }, [state]);

  /**
   * Toggle section expansion
   */
  const toggleSection = useCallback((section: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  }, []);

  /**
   * Handle basic field changes
   */
  const handleFieldChange = useCallback((field: keyof StoryState, value: any) => {
    onUpdateState({ [field]: value } as Partial<StoryState>);
  }, [onUpdateState]);

  /**
   * Handle zone change
   */
  const handleZoneChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newZone = e.target.value || null;
    setZone(newZone);
    handleFieldChange('zone', newZone);
  }, [handleFieldChange]);

  /**
   * Handle label change with debounce
   */
  const handleLabelChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setLabel(e.target.value);
  }, []);

  const handleLabelBlur = useCallback(() => {
    handleFieldChange('label', label);
  }, [label, handleFieldChange]);

  /**
   * Handle description change
   */
  const handleDescriptionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDescription(e.target.value);
  }, []);

  const handleDescriptionBlur = useCallback(() => {
    handleFieldChange('description', description);
  }, [description, handleFieldChange]);

  /**
   * Handle notes change
   */
  const handleNotesChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setNotes(e.target.value);
  }, []);

  const handleNotesBlur = useCallback(() => {
    handleFieldChange('notes', notes);
  }, [notes, handleFieldChange]);

  /**
   * Capture player position from viewport
   */
  const handleCapturePosition = useCallback(() => {
    if (onCapturePosition) {
      const pos = onCapturePosition();
      if (pos) {
        setPlayerPos(pos);
        handleFieldChange('playerPosition', pos);
      }
    } else {
      // Default position for demo
      const defaultPos: PlayerPosition = {
        x: 0,
        y: 1.6,
        z: 0,
      };
      setPlayerPos(defaultPos);
      handleFieldChange('playerPosition', defaultPos);
    }
  }, [onCapturePosition, handleFieldChange]);

  /**
   * Update content field
   */
  const handleContentChange = useCallback((field: keyof StateContent, value: any) => {
    const updatedContent = { ...content, [field]: value };
    setContent(updatedContent);
    handleFieldChange('content', updatedContent);
  }, [content, handleFieldChange]);

  /**
   * Render transition item
   */
  const renderTransition = useCallback((transition: Transition) => {
    const getTriggerIcon = (trigger: TriggerType): string => {
      switch (trigger) {
        case 'onComplete': return '✓';
        case 'onChoice': return '◆';
        case 'onTimeout': return '⏱';
        case 'onProximity': return '◎';
        case 'onInteract': return '👆';
        case 'onState': return '→';
        case 'custom': return '⚙';
        default: return '?';
      }
    };

    return (
      <div key={transition.id} className="inspector-transition-item">
        <div className="transition-header">
          <span className="transition-icon">{getTriggerIcon(transition.trigger)}</span>
          <span className="transition-label">{transition.label}</span>
        </div>
        <div className="transition-details">
          <div className="transition-detail">
            <span className="detail-label">Trigger:</span>
            <span className="detail-value">{transition.trigger}</span>
          </div>
          <div className="transition-detail">
            <span className="detail-label">Target:</span>
            <span className="detail-value">{transition.to}</span>
          </div>
        </div>
        <div className="transition-actions">
          <button className="btn-small btn-secondary">Edit</button>
          <button className="btn-small btn-danger">Delete</button>
        </div>
      </div>
    );
  }, []);

  /**
   * Render content field
   */
  const renderContentField = useCallback((
    label: string,
    field: keyof StateContent,
    value: string | undefined,
    icon: string,
    placeholder: string
  ) => {
    return (
      <div className="inspector-content-field">
        <label className="field-label">
          <span className="field-icon">{icon}</span>
          {label}
        </label>
        <div className="field-input-group">
          <input
            type="text"
            className="field-input"
            value={value || ''}
            onChange={(e) => handleContentChange(field, e.target.value || undefined)}
            placeholder={placeholder}
          />
          {value && (
            <button className="btn-small btn-secondary">Preview</button>
          )}
        </div>
      </div>
    );
  }, [handleContentChange]);

  /**
   * Render collapsible section
   */
  const renderSection = useCallback((
    id: string,
    title: string,
    icon: string,
    children: React.ReactNode
  ) => {
    const isExpanded = expandedSections.has(id);
    return (
      <div className="inspector-section">
        <div
          className={`inspector-section-header ${isExpanded ? 'expanded' : ''}`}
          onClick={() => toggleSection(id)}
        >
          <span className="section-icon">{icon}</span>
          <span className="section-title">{title}</span>
          <span className="section-toggle">{isExpanded ? '▼' : '▶'}</span>
        </div>
        {isExpanded && (
          <div className="inspector-section-content">
            {children}
          </div>
        )}
      </div>
    );
  }, [expandedSections, toggleSection]);

  if (!state) {
    return (
      <div className="state-inspector state-inspector-empty">
        <div className="empty-state">
          <span className="empty-icon">📋</span>
          <p>Select a state to inspect and edit</p>
        </div>
      </div>
    );
  }

  return (
    <div className="state-inspector">
      {/* State Header */}
      <div className="inspector-header">
        <div className="header-title">
          <span className="state-value-badge">{state.value}</span>
          <h3 className="header-state-name">{state.id}</h3>
        </div>
        <div className="header-actions">
          <button
            className="btn-small btn-secondary"
            onClick={() => toggleSection('basic')}
            title="Edit basic properties"
          >
            ✏️ Edit
          </button>
        </div>
      </div>

      {/* Basic Properties Section */}
      {renderSection('basic', 'Basic Properties', '📝', (
        <div className="inspector-fields">
          <div className="inspector-field">
            <label className="field-label">Label</label>
            <input
              type="text"
              className="field-input"
              value={label}
              onChange={handleLabelChange}
              onBlur={handleLabelBlur}
              placeholder="State label"
            />
          </div>
          <div className="inspector-field">
            <label className="field-label">Description</label>
            <textarea
              className="field-textarea"
              value={description}
              onChange={handleDescriptionChange}
              onBlur={handleDescriptionBlur}
              placeholder="State description"
              rows={2}
            />
          </div>
          <div className="inspector-field">
            <label className="field-label">Notes</label>
            <textarea
              className="field-textarea"
              value={notes}
              onChange={handleNotesChange}
              onBlur={handleNotesBlur}
              placeholder="Developer notes"
              rows={3}
            />
          </div>
        </div>
      ))}

      {/* Zone Assignment Section */}
      {renderSection('zone', 'Zone Assignment', '📍', (
        <div className="inspector-fields">
          <div className="inspector-field">
            <label className="field-label">Zone</label>
            <div className="field-input-group">
              <select
                className="field-select"
                value={zone}
                onChange={handleZoneChange}
              >
                <option value="">No zone assigned</option>
                {availableZones.map((z) => (
                  <option key={z.id} value={z.id}>
                    {z.name}
                  </option>
                ))}
              </select>
              {zone && onJumpToZone && (
                <button
                  className="btn-small btn-secondary"
                  onClick={() => onJumpToZone(zone)}
                  title="Jump to zone in 3D viewport"
                >
                  Jump Zone
                </button>
              )}
            </div>
          </div>

          {/* Player Position */}
          <div className="inspector-field">
            <label className="field-label">Player Position</label>
            {playerPos ? (
              <div className="player-position-display">
                <div className="position-coords">
                  <span className="coord">x: {playerPos.x.toFixed(2)}</span>
                  <span className="coord">y: {playerPos.y.toFixed(2)}</span>
                  <span className="coord">z: {playerPos.z.toFixed(2)}</span>
                </div>
                <button
                  className="btn-small btn-secondary"
                  onClick={handleCapturePosition}
                  title="Capture current player position from viewport"
                >
                  🎯 Capture View
                </button>
              </div>
            ) : (
              <button
                className="btn-small btn-secondary"
                onClick={handleCapturePosition}
              >
                🎯 Capture from Viewport
              </button>
            )}
          </div>
        </div>
      ))}

      {/* Content Attachments Section */}
      {renderSection('content', 'Content Attachments', '📎', (
        <div className="inspector-content">
          {renderContentField('Video', 'video', content.video, '📹', 'video-file.mp4')}
          {renderContentField('Dialog', 'dialog', content.dialog, '💬', 'dialog_tree_id')}
          {renderContentField('Music', 'music', content.music, '🎵', 'music_track_id')}
          {renderContentField('Camera Animation', 'cameraAnimation', content.cameraAnimation, '🎬', 'camera_anim_id')}
        </div>
      ))}

      {/* Transitions Section */}
      {renderSection('transitions', `Transitions (${state.transitions.length})`, '↗', (
        <div className="inspector-transitions">
          {state.transitions.length === 0 ? (
            <p className="no-transitions">No outgoing transitions</p>
          ) : (
            <div className="transitions-list">
              {state.transitions.map(renderTransition)}
            </div>
          )}
          <button className="btn-small btn-primary">+ Add Transition</button>
        </div>
      ))}

      {/* Action Buttons */}
      <div className="inspector-actions">
        <button className="btn btn-primary" onClick={onTestState}>
          ▶ Test State
        </button>
        <button className="btn btn-secondary" onClick={onDuplicateState}>
          📋 Duplicate
        </button>
        <button className="btn btn-danger" onClick={onDeleteState}>
          🗑 Delete
        </button>
      </div>
    </div>
  );
};

export default StateInspector;
