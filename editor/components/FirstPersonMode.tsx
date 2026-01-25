import React, { useState, useEffect, useCallback } from 'react';
import EditorManager from '../core/EditorManager';
import SimpleFirstPersonControls from '../core/SimpleFirstPersonControls';
import TransformGizmoManager from '../core/TransformGizmoManager';
import './FirstPersonMode.css';

/**
 * FirstPersonMode Toggle Component
 *
 * Provides EXACT game-like first-person controls:
 * - WASD movement (forward, left, backward, right)
 * - Mouse look (pointer lock)
 * - Shift to sprint
 * - Headbob while walking
 * - Camera smoothing (0.15) like the game
 * - Ground detection for walking
 */
interface FirstPersonModeProps {
    editorManager: EditorManager;
}

const FirstPersonMode: React.FC<FirstPersonModeProps> = ({ editorManager }) => {
    const [isEnabled, setIsEnabled] = useState(false);
    const [isPointerLocked, setIsPointerLocked] = useState(false);

    // FirstPersonControls instance
    const controlsRef = React.useRef<SimpleFirstPersonControls | null>(null);

    /**
     * Initialize FirstPersonControls with GAME-EXACT defaults
     */
    useEffect(() => {
        if (!editorManager.camera || !editorManager.renderer) {
            console.log('FirstPersonMode: Waiting for camera/renderer...');
            return;
        }

        console.log('FirstPersonMode: Initializing with game-exact settings...');

        const controls = new SimpleFirstPersonControls(
            editorManager.camera,
            editorManager.renderer.domElement,
            {
                // EXACT game defaults from CharacterController
                moveSpeed: 2.5,              // Game's baseSpeed
                sprintMultiplier: 1.75,       // Game's sprintMultiplier
                mouseSensitivity: 0.0025,     // Game's mouseSensitivity
                cameraHeight: 0.9,            // Game's cameraHeight
                cameraSmoothingFactor: 0.15,  // Game's cameraSmoothingFactor
                orbitControls: editorManager.orbitControls,
                scene: editorManager.scene,
            }
        );

        controlsRef.current = controls;

        return () => {
            controls.dispose();
        };
    }, [editorManager]);

    /**
     * Enable first-person mode
     */
    const handleEnable = useCallback(() => {
        if (controlsRef.current) {
            // Disable gizmo keyboard shortcuts to prevent WASD conflicts
            const gizmoManager = TransformGizmoManager.getInstance();
            gizmoManager.setKeyboardShortcutsEnabled(false);

            controlsRef.current.enable();
            setIsEnabled(true);
        }
    }, []);

    /**
     * Disable first-person mode
     */
    const handleDisable = useCallback(() => {
        if (controlsRef.current) {
            controlsRef.current.disable();
            setIsEnabled(false);
            setIsPointerLocked(false);

            // Re-enable gizmo keyboard shortcuts
            const gizmoManager = TransformGizmoManager.getInstance();
            gizmoManager.setKeyboardShortcutsEnabled(true);
        }
    }, []);

    /**
     * Toggle first-person mode
     */
    const handleToggle = useCallback(() => {
        if (isEnabled) {
            handleDisable();
        } else {
            handleEnable();
        }
    }, [isEnabled, handleEnable, handleDisable]);

    /**
     * Handle keyboard shortcut (F9)
     */
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // F9 to toggle first-person mode
            if (e.key === 'F9') {
                e.preventDefault();
                handleToggle();
            }

            // Escape to exit first-person mode
            if (e.key === 'Escape' && isEnabled) {
                handleDisable();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isEnabled, handleToggle, handleDisable]);

    /**
     * Update pointer lock state
     */
    useEffect(() => {
        const checkPointerLock = () => {
            const locked = document.pointerLockElement === editorManager.renderer?.domElement;
            setIsPointerLocked(locked);
        };

        // Check periodically
        const interval = setInterval(checkPointerLock, 100);

        return () => clearInterval(interval);
    }, [editorManager]);

    return (
        <div className={`firstperson-mode ${isEnabled ? 'enabled' : ''} ${isPointerLocked ? 'locked' : ''}`}>
            {/* Toggle Button */}
            <button
                className={`fp-toggle ${isEnabled ? 'active' : ''}`}
                onClick={handleToggle}
                title="Toggle First-Person Mode (F9)"
            >
                <span className="fp-icon">{isEnabled ? '👁️' : '🎮'}</span>
                <span className="fp-label">{isEnabled ? 'FPS' : 'Orbit'}</span>
            </button>

            {/* Status indicator */}
            {isEnabled && (
                <div className="fp-status">
                    <span className={`fp-indicator ${isPointerLocked ? 'locked' : ''}`} />
                    <span className="fp-controls-hint">
                        {isPointerLocked
                            ? 'WASD to move • Shift to sprint • Mouse to look'
                            : 'Click viewport to enable mouse look'}
                    </span>
                </div>
            )}

            {/* Instructions */}
            {isEnabled && !isPointerLocked && (
                <div style={{
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    background: 'rgba(0, 0, 0, 0.8)',
                    color: '#fff',
                    padding: '20px 40px',
                    borderRadius: '8px',
                    zIndex: 1000,
                    pointerEvents: 'none'
                }}>
                    <h3 style={{ margin: '0 0 10px 0' }}>Click to enable mouse look</h3>
                    <p style={{ margin: 0 }}>Press ESC or F9 to exit first-person mode</p>
                </div>
            )}
        </div>
    );
};

export default FirstPersonMode;
