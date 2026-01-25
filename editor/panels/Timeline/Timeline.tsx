import React, { useState, useEffect, useRef, useCallback } from 'react';
import EditorManager from '../../core/EditorManager';
import './Timeline.css';

/**
 * Keyframe interface for animation tracks
 */
interface Keyframe {
    id: string;
    time: number;
    value: any;
    easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
}

/**
 * Animation track interface
 */
interface AnimationTrack {
    id: string;
    name: string;
    type: 'camera-position' | 'camera-lookAt' | 'camera-fov' |
          'object-position' | 'object-rotation' | 'object-scale' |
          'object-visibility' | 'object-opacity';
    objectId?: string; // For object tracks
    keyframes: Keyframe[];
    color: string;
    enabled: boolean;
}

/**
 * Timeline State
 */
interface TimelineState {
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    playbackSpeed: number;
    selectedTrack: string | null;
    selectedKeyframe: string | null;
    loop: boolean;
}

/**
 * Timeline Panel - Animation timeline with keyframe editing
 *
 * Features:
 * - Timeline scrubbing with play/pause controls
 * - Time display (current/total)
 * - Multiple animation tracks
 * - Visual keyframe representation
 * - Easing curve selection
 * - Camera animation tracks (position, lookAt, FOV)
 * - Object animation tracks (position, rotation, scale, visibility, opacity)
 */
interface TimelineProps {
    editorManager: EditorManager;
}

const Timeline: React.FC<TimelineProps> = ({ editorManager }) => {
    // Timeline state
    const [state, setState] = useState<TimelineState>({
        isPlaying: false,
        currentTime: 0,
        duration: 10,
        playbackSpeed: 1,
        selectedTrack: null,
        selectedKeyframe: null,
        loop: false
    });

    // Animation tracks
    const [tracks, setTracks] = useState<AnimationTrack[]>([
        {
            id: 'camera-position',
            name: 'Camera Position',
            type: 'camera-position',
            keyframes: [],
            color: '#4a9eff',
            enabled: true
        },
        {
            id: 'camera-lookat',
            name: 'Camera Look At',
            type: 'camera-lookAt',
            keyframes: [],
            color: '#ff6b6b',
            enabled: true
        },
        {
            id: 'camera-fov',
            name: 'Camera FOV',
            type: 'camera-fov',
            keyframes: [],
            color: '#51cf66',
            enabled: true
        }
    ]);

    // UI state
    const [zoom, setZoom] = useState(50); // Pixels per second
    const [showAddTrackMenu, setShowAddTrackMenu] = useState(false);

    // Animation frame ref
    const animationRef = useRef<number>();
    const lastTimeRef = useRef<number>();
    const timelineRef = useRef<HTMLDivElement>(null);
    const isScrubbingRef = useRef(false);

    /**
     * Format time for display
     */
    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const frames = Math.floor((seconds % 1) * 30);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
    };

    /**
     * Play/pause toggle
     */
    const togglePlay = useCallback(() => {
        setState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
    }, []);

    /**
     * Stop playback and reset to beginning
     */
    const stop = useCallback(() => {
        setState(prev => ({ ...prev, isPlaying: false, currentTime: 0 }));
    }, []);

    /**
     * Jump to start
     */
    const goToStart = useCallback(() => {
        setState(prev => ({ ...prev, currentTime: 0 }));
    }, []);

    /**
     * Jump to end
     */
    const goToEnd = useCallback(() => {
        setState(prev => ({ ...prev, currentTime: prev.duration }));
    }, []);

    /**
     * Set current time
     */
    const setCurrentTime = useCallback((time: number) => {
        setState(prev => ({ ...prev, currentTime: Math.max(0, Math.min(time, prev.duration)) }));
    }, []);

    /**
     * Set duration
     */
    const setDuration = useCallback((duration: number) => {
        setState(prev => ({ ...prev, duration: Math.max(1, duration) }));
    }, []);

    /**
     * Add keyframe at current time
     */
    const addKeyframe = useCallback((trackId: string) => {
        const track = tracks.find(t => t.id === trackId);
        if (!track) return;

        let value: any = null;

        // Get current value based on track type
        switch (track.type) {
            case 'camera-position':
                value = {
                    x: editorManager.camera.position.x,
                    y: editorManager.camera.position.y,
                    z: editorManager.camera.position.z
                };
                break;
            case 'camera-fov':
                value = editorManager.camera.fov;
                break;
            case 'camera-lookat':
                const lookAt = new THREE.Vector3();
                editorManager.camera.getWorldDirection(lookAt);
                value = { x: lookAt.x, y: lookAt.y, z: lookAt.z };
                break;
        }

        const newKeyframe: Keyframe = {
            id: `kf-${Date.now()}-${Math.random()}`,
            time: state.currentTime,
            value,
            easing: 'linear'
        };

        setTracks(prev => prev.map(t => {
            if (t.id === trackId) {
                // Remove existing keyframe at same time
                const filtered = t.keyframes.filter(kf => Math.abs(kf.time - state.currentTime) > 0.01);
                return {
                    ...t,
                    keyframes: [...filtered, newKeyframe].sort((a, b) => a.time - b.time)
                };
            }
            return t;
        }));
    }, [tracks, state.currentTime, editorManager]);

    /**
     * Delete selected keyframe
     */
    const deleteKeyframe = useCallback(() => {
        if (!state.selectedKeyframe || !state.selectedTrack) return;

        setTracks(prev => prev.map(t => {
            if (t.id === state.selectedTrack) {
                return {
                    ...t,
                    keyframes: t.keyframes.filter(kf => kf.id !== state.selectedKeyframe)
                };
            }
            return t;
        }));

        setState(prev => ({ ...prev, selectedKeyframe: null }));
    }, [state.selectedKeyframe, state.selectedTrack]);

    /**
     * Update keyframe easing
     */
    const updateKeyframeEasing = useCallback((keyframeId: string, easing: Keyframe['easing']) => {
        setTracks(prev => prev.map(t => ({
            ...t,
            keyframes: t.keyframes.map(kf =>
                kf.id === keyframeId ? { ...kf, easing } : kf
            )
        })));
    }, []);

    /**
     * Add object track
     */
    const addObjectTrack = useCallback((objectId: string, trackType: AnimationTrack['type']) => {
        const selectedObject = editorManager.selectedObject;
        if (!selectedObject) return;

        const newTrack: AnimationTrack = {
            id: `object-${objectId}-${trackType}-${Date.now()}`,
            name: `${selectedObject.name || 'Object'} - ${trackType.replace('object-', '').toUpperCase()}`,
            type: trackType,
            objectId,
            keyframes: [],
            color: `#${Math.floor(Math.random() * 0x1000000).toString(16).padStart(6, '0')}`,
            enabled: true
        };

        setTracks(prev => [...prev, newTrack]);
        setShowAddTrackMenu(false);
    }, [editorManager]);

    /**
     * Delete track
     */
    const deleteTrack = useCallback((trackId: string) => {
        setTracks(prev => prev.filter(t => t.id !== trackId));
        if (state.selectedTrack === trackId) {
            setState(prev => ({ ...prev, selectedTrack: null }));
        }
    }, [state.selectedTrack]);

    /**
     * Handle timeline scrubbing
     */
    const handleTimelineMouseDown = (e: React.MouseEvent) => {
        if (!timelineRef.current) return;

        isScrubbingRef.current = true;
        updateTimeFromMouse(e);
    };

    const updateTimeFromMouse = (e: React.MouseEvent | MouseEvent) => {
        if (!timelineRef.current) return;

        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const time = (x / zoom);

        setCurrentTime(time);
    };

    /**
     * Handle keyframe drag
     */
    const handleKeyframeDrag = useCallback((trackId: string, keyframeId: string, newTime: number) => {
        setTracks(prev => prev.map(t => {
            if (t.id === trackId) {
                return {
                    ...t,
                    keyframes: t.keyframes.map(kf =>
                        kf.id === keyframeId ? { ...kf, time: Math.max(0, newTime) } : kf
                    ).sort((a, b) => a.time - b.time)
                };
            }
            return t;
        }));
    }, []);

    /**
     * Evaluate animation at current time
     */
    useEffect(() => {
        const evaluateAnimation = () => {
            tracks.forEach(track => {
                if (!track.enabled) return;

                // Find keyframes surrounding current time
                const beforeKF = [...track.keyframes]
                    .filter(kf => kf.time <= state.currentTime)
                    .sort((a, b) => b.time - a.time)[0];

                const afterKF = [...track.keyframes]
                    .filter(kf => kf.time > state.currentTime)
                    .sort((a, b) => a.time - b.time)[0];

                if (!beforeKF && !afterKF) return;

                let value: any;

                if (beforeKF && afterKF) {
                    // Interpolate between keyframes
                    const t = (state.currentTime - beforeKF.time) / (afterKF.time - beforeKF.time);
                    const easedT = applyEasing(t, beforeKF.easing);
                    value = interpolateValues(beforeKF.value, afterKF.value, easedT);
                } else if (beforeKF) {
                    value = beforeKF.value;
                } else {
                    value = afterKF.value;
                }

                // Apply value based on track type
                applyTrackValue(track, value);
            });
        };

        evaluateAnimation();
    }, [state.currentTime, tracks]);

    /**
     * Apply easing function
     */
    const applyEasing = (t: number, easing: Keyframe['easing']): number => {
        switch (easing) {
            case 'linear': return t;
            case 'ease-in': return t * t;
            case 'ease-out': return t * (2 - t);
            case 'ease-in-out': return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
            default: return t;
        }
    };

    /**
     * Interpolate values based on type
     */
    const interpolateValues = (from: any, to: any, t: number): any => {
        if (typeof from === 'number' && typeof to === 'number') {
            return from + (to - from) * t;
        }

        if (typeof from === 'object' && typeof to === 'object') {
            const result: any = {};
            for (const key in from) {
                if (key in to) {
                    result[key] = from[key] + (to[key] - from[key]) * t;
                }
            }
            return result;
        }

        return to;
    };

    /**
     * Apply track value to scene
     */
    const applyTrackValue = (track: AnimationTrack, value: any) => {
        switch (track.type) {
            case 'camera-position':
                if (value) {
                    editorManager.camera.position.set(value.x, value.y, value.z);
                }
                break;
            case 'camera-fov':
                if (typeof value === 'number') {
                    editorManager.camera.fov = value;
                    editorManager.camera.updateProjectionMatrix();
                }
                break;
            case 'camera-lookat':
                if (value) {
                    // Update lookAt target (simplified)
                    // In a real implementation, you'd track a target object
                }
                break;
            case 'object-position':
                if (track.objectId && value) {
                    const obj = editorManager.scene.getObjectByProperty('uuid', track.objectId);
                    if (obj) {
                        obj.position.set(value.x, value.y, value.z);
                    }
                }
                break;
            case 'object-rotation':
                if (track.objectId && value) {
                    const obj = editorManager.scene.getObjectByProperty('uuid', track.objectId);
                    if (obj) {
                        obj.rotation.set(value.x, value.y, value.z);
                    }
                }
                break;
            case 'object-scale':
                if (track.objectId && value) {
                    const obj = editorManager.scene.getObjectByProperty('uuid', track.objectId);
                    if (obj) {
                        if (typeof value === 'number') {
                            obj.scale.setScalar(value);
                        } else {
                            obj.scale.set(value.x, value.y, value.z);
                        }
                    }
                }
                break;
        }
    };

    /**
     * Animation loop for playback
     */
    useEffect(() => {
        if (!state.isPlaying) {
            lastTimeRef.current = undefined;
            return;
        }

        const animate = (timestamp: number) => {
            if (lastTimeRef.current === undefined) {
                lastTimeRef.current = timestamp;
            }

            const deltaTime = (timestamp - lastTimeRef.current) / 1000;
            lastTimeRef.current = timestamp;

            setState(prev => {
                const newTime = prev.currentTime + deltaTime * prev.playbackSpeed;

                if (newTime >= prev.duration) {
                    if (prev.loop) {
                        return { ...prev, currentTime: 0 };
                    } else {
                        return { ...prev, isPlaying: false, currentTime: prev.duration };
                    }
                }

                return { ...prev, currentTime: newTime };
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [state.isPlaying, state.playbackSpeed, state.duration, state.loop]);

    /**
     * Handle keyboard shortcuts
     */
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Space - Play/Pause
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                togglePlay();
            }

            // Delete - Remove selected keyframe
            if (e.code === 'Delete' && state.selectedKeyframe) {
                deleteKeyframe();
            }

            // Home - Go to start
            if (e.code === 'Home') {
                goToStart();
            }

            // End - Go to end
            if (e.code === 'End') {
                goToEnd();
            }

            // K - Add keyframe
            if (e.key === 'k' && state.selectedTrack) {
                addKeyframe(state.selectedTrack);
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        // Mouse move for scrubbing
        const handleMouseMove = (e: MouseEvent) => {
            if (isScrubbingRef.current) {
                updateTimeFromMouse(e);
            }
        };

        const handleMouseUp = () => {
            isScrubbingRef.current = false;
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [togglePlay, deleteKeyframe, goToStart, goToEnd, addKeyframe, state.selectedTrack, state.selectedKeyframe]);

    /**
     * Render a single track
     */
    const renderTrack = (track: AnimationTrack) => {
        const isSelected = state.selectedTrack === track.id;

        return (
            <div
                key={track.id}
                className={`timeline-track ${isSelected ? 'selected' : ''} ${!track.enabled ? 'disabled' : ''}`}
                onClick={() => setState(prev => ({ ...prev, selectedTrack: track.id }))}
            >
                {/* Track header */}
                <div className="track-header">
                    <div className="track-info">
                        <input
                            type="checkbox"
                            checked={track.enabled}
                            onChange={(e) => {
                                e.stopPropagation();
                                setTracks(prev => prev.map(t =>
                                    t.id === track.id ? { ...t, enabled: e.target.checked } : t
                                ));
                            }}
                            className="track-enabled"
                        />
                        <span
                            className="track-color"
                            style={{ backgroundColor: track.color }}
                        />
                        <span className="track-name">{track.name}</span>
                    </div>
                    <button
                        className="track-delete"
                        onClick={(e) => {
                            e.stopPropagation();
                            deleteTrack(track.id);
                        }}
                    >
                        ×
                    </button>
                </div>

                {/* Track timeline */}
                <div className="track-timeline" ref={timelineRef}>
                    {/* Keyframes */}
                    {track.keyframes.map(kf => (
                        <div
                            key={kf.id}
                            className={`keyframe ${state.selectedKeyframe === kf.id ? 'selected' : ''}`}
                            style={{
                                left: `${kf.time * zoom}px`,
                                backgroundColor: track.color
                            }}
                            onClick={(e) => {
                                e.stopPropagation();
                                setState(prev => ({ ...prev, selectedKeyframe: kf.id }));
                            }}
                            onDrag={(e) => {
                                // Handle drag if needed
                            }}
                            title={`${track.name} - ${formatTime(kf.time)}`}
                        >
                            <span className="keyframe-marker">◆</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="timeline-container">
            {/* Timeline header */}
            <div className="timeline-header">
                <div className="timeline-controls">
                    <button
                        className={`control-button ${state.isPlaying ? 'active' : ''}`}
                        onClick={togglePlay}
                        title="Play/Pause (Space)"
                    >
                        {state.isPlaying ? '⏸' : '▶'}
                    </button>
                    <button
                        className="control-button"
                        onClick={stop}
                        title="Stop"
                    >
                        ⏹
                    </button>
                    <button
                        className="control-button"
                        onClick={goToStart}
                        title="Go to Start (Home)"
                    >
                        ⏮
                    </button>
                    <button
                        className="control-button"
                        onClick={goToEnd}
                        title="Go to End (End)"
                    >
                        ⏭
                    </button>
                </div>

                <div className="timeline-time">
                    <span className="time-current">{formatTime(state.currentTime)}</span>
                    <span className="time-separator">/</span>
                    <span className="time-total">{formatTime(state.duration)}</span>
                </div>

                <div className="timeline-settings">
                    <label className="setting-label">
                        Speed:
                        <select
                            value={state.playbackSpeed}
                            onChange={(e) => setState(prev => ({ ...prev, playbackSpeed: parseFloat(e.target.value) }))}
                            className="setting-select"
                        >
                            <option value={0.25}>0.25x</option>
                            <option value={0.5}>0.5x</option>
                            <option value={1}>1x</option>
                            <option value={2}>2x</option>
                        </select>
                    </label>

                    <label className="setting-label">
                        <input
                            type="checkbox"
                            checked={state.loop}
                            onChange={(e) => setState(prev => ({ ...prev, loop: e.target.checked }))}
                        />
                        Loop
                    </label>
                </div>

                <div className="timeline-actions">
                    {state.selectedTrack && (
                        <button
                            className="action-button"
                            onClick={() => state.selectedTrack && addKeyframe(state.selectedTrack)}
                            title="Add Keyframe (K)"
                        >
                            + Keyframe
                        </button>
                    )}
                    <button
                        className="action-button"
                        onClick={() => setShowAddTrackMenu(!showAddTrackMenu)}
                    >
                        + Track
                    </button>
                </div>
            </div>

            {/* Timeline ruler */}
            <div className="timeline-ruler-container">
                <div
                    className="timeline-ruler"
                    style={{ width: `${state.duration * zoom}px` }}
                    onMouseDown={handleTimelineMouseDown}
                >
                    {/* Time markers */}
                    {Array.from({ length: Math.ceil(state.duration) + 1 }, (_, i) => (
                        <div
                            key={i}
                            className="ruler-marker"
                            style={{ left: `${i * zoom}px` }}
                        >
                            <span className="ruler-label">{formatTime(i)}</span>
                        </div>
                    ))}

                    {/* Playhead */}
                    <div
                        className="playhead"
                        style={{ left: `${state.currentTime * zoom}px` }}
                    >
                        <div className="playhead-line" />
                        <div className="playhead-handle" />
                    </div>
                </div>
            </div>

            {/* Add track menu */}
            {showAddTrackMenu && (
                <div className="add-track-menu">
                    <div className="menu-section">
                        <h4>Camera Tracks</h4>
                        {tracks.filter(t => t.type.startsWith('camera')).length === 0 && (
                            <button onClick={() => {
                                setTracks(prev => [...prev, {
                                    id: `camera-position-${Date.now()}`,
                                    name: 'Camera Position',
                                    type: 'camera-position',
                                    keyframes: [],
                                    color: '#4a9eff',
                                    enabled: true
                                }]);
                                setShowAddTrackMenu(false);
                            }}>Position</button>
                        )}
                    </div>
                    <div className="menu-section">
                        <h4>Object Tracks</h4>
                        {editorManager.selectedObject ? (
                            <>
                                <button onClick={() => addObjectTrack(editorManager.selectedObject!.uuid, 'object-position')}>
                                    Position
                                </button>
                                <button onClick={() => addObjectTrack(editorManager.selectedObject!.uuid, 'object-rotation')}>
                                    Rotation
                                </button>
                                <button onClick={() => addObjectTrack(editorManager.selectedObject!.uuid, 'object-scale')}>
                                    Scale
                                </button>
                            </>
                        ) : (
                            <small>Select an object first</small>
                        )}
                    </div>
                </div>
            )}

            {/* Tracks */}
            <div className="timeline-tracks">
                {tracks.length === 0 ? (
                    <div className="timeline-empty">
                        <p>No animation tracks</p>
                        <p className="hint">Click "+ Track" to add a track</p>
                    </div>
                ) : (
                    tracks.map(renderTrack)
                )}
            </div>

            {/* Keyframe properties */}
            {state.selectedKeyframe && (
                <div className="keyframe-properties">
                    <h4>Keyframe Properties</h4>
                    {tracks.flatMap(t => t.keyframes).find(kf => kf.id === state.selectedKeyframe) && (
                        <>
                            <div className="property-row">
                                <label>Time:</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={tracks.flatMap(t => t.keyframes).find(kf => kf.id === state.selectedKeyframe)?.time || 0}
                                    onChange={(e) => {
                                        const track = tracks.find(t => t.keyframes.some(kf => kf.id === state.selectedKeyframe));
                                        if (track) {
                                            handleKeyframeDrag(track.id, state.selectedKeyframe!, parseFloat(e.target.value));
                                        }
                                    }}
                                />
                            </div>
                            <div className="property-row">
                                <label>Easing:</label>
                                <select
                                    value={tracks.flatMap(t => t.keyframes).find(kf => kf.id === state.selectedKeyframe)?.easing || 'linear'}
                                    onChange={(e) => updateKeyframeEasing(state.selectedKeyframe!, e.target.value as Keyframe['easing'])}
                                >
                                    <option value="linear">Linear</option>
                                    <option value="ease-in">Ease In</option>
                                    <option value="ease-out">Ease Out</option>
                                    <option value="ease-in-out">Ease In-Out</option>
                                </select>
                            </div>
                            <button
                                className="delete-keyframe-btn"
                                onClick={deleteKeyframe}
                            >
                                Delete Keyframe
                            </button>
                        </>
                    )}
                </div>
            )}

            {/* Timeline footer */}
            <div className="timeline-footer">
                <div className="zoom-control">
                    <button onClick={() => setZoom(Math.max(10, zoom - 10))}>-</button>
                    <span>Zoom: {zoom}px/s</span>
                    <button onClick={() => setZoom(Math.min(200, zoom + 10))}>+</button>
                </div>
                <div className="duration-control">
                    <label>Duration:</label>
                    <input
                        type="number"
                        min="1"
                        step="0.1"
                        value={state.duration}
                        onChange={(e) => setDuration(parseFloat(e.target.value))}
                    />
                    <span>s</span>
                </div>
            </div>
        </div>
    );
};

export default Timeline;
