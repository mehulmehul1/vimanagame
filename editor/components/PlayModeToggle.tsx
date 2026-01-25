import React, { useState, useEffect } from 'react';
import EditorManager from '../core/EditorManager';
import './PlayModeToggle.css';

/**
 * PlayModeToggle - Toggle button for Play/Edit mode switching
 *
 * Features:
 * - Play/Edit mode toggle button
 * - Visual indicator (badge/color)
 * - Keyboard shortcut (Ctrl+P or Space)
 * - Mode-specific tooltips
 * - Disabled state during mode transitions
 */
interface PlayModeToggleProps {
    editorManager: EditorManager;
    onModeChanged?: (isPlaying: boolean) => void;
}

const PlayModeToggle: React.FC<PlayModeToggleProps> = ({ editorManager, onModeChanged }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);

    useEffect(() => {
        // Listen to play mode changes from EditorManager
        const handlePlayModeEntered = () => {
            setIsPlaying(true);
            setIsTransitioning(false);
            if (onModeChanged) {
                onModeChanged(true);
            }
        };

        const handlePlayModeExited = () => {
            setIsPlaying(false);
            setIsTransitioning(false);
            if (onModeChanged) {
                onModeChanged(false);
            }
        };

        editorManager.on('playModeEntered', handlePlayModeEntered);
        editorManager.on('playModeExited', handlePlayModeExited);

        // Listen for keyboard shortcuts
        const handleKeyDown = (event: KeyboardEvent) => {
            // Ctrl+P or Space to toggle play mode
            if ((event.ctrlKey && event.key === 'p') || event.key === ' ') {
                event.preventDefault();
                togglePlayMode();
            }
        };

        document.addEventListener('keydown', handleKeyDown);

        return () => {
            editorManager.off('playModeEntered', handlePlayModeEntered);
            editorManager.off('playModeExited', handlePlayModeExited);
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [editorManager, onModeChanged]);

    /**
     * Toggle between play and edit mode
     */
    const togglePlayMode = async () => {
        if (isTransitioning) return;

        setIsTransitioning(true);

        if (isPlaying) {
            console.log('PlayModeToggle: Exiting play mode');
            editorManager.exitPlayMode();
        } else {
            console.log('PlayModeToggle: Entering play mode');
            editorManager.enterPlayMode();
        }
    };

    /**
     * Get button text based on mode
     */
    const getButtonText = () => {
        if (isTransitioning) {
            return isPlaying ? 'Stopping...' : 'Starting...';
        }
        return isPlaying ? '⏹ Stop' : '▶ Play';
    };

    /**
     * Get button title/tooltip
     */
    const getButtonTitle = () => {
        if (isTransitioning) {
            return 'Please wait...';
        }
        return isPlaying
            ? 'Exit play mode (Ctrl+P or Space)'
            : 'Enter play mode (Ctrl+P or Space)';
    };

    return (
        <div className={`play-mode-toggle ${isPlaying ? 'playing' : 'editing'}`}>
            <button
                className={`play-mode-button ${isPlaying ? 'active' : ''} ${isTransitioning ? 'transitioning' : ''}`}
                onClick={togglePlayMode}
                disabled={isTransitioning}
                title={getButtonTitle()}
            >
                <span className="play-mode-icon">{isPlaying ? '⏹' : '▶'}</span>
                <span className="play-mode-text">{getButtonText()}</span>
            </button>

            {/* Mode badge */}
            <div className={`mode-badge ${isPlaying ? 'badge-playing' : 'badge-editing'}`}>
                {isPlaying ? 'PLAY' : 'EDIT'}
            </div>

            {/* Keyboard shortcut hint */}
            <div className="shortcut-hint">
                Ctrl+P
            </div>
        </div>
    );
};

export default PlayModeToggle;
