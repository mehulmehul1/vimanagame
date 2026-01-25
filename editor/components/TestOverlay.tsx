/**
 * TestOverlay.tsx - Test Mode Overlay Component
 * =============================================================================
 *
 * Overlay shown during state testing in the game.
 * Features:
 * - Displays current state name and value
 * - "Return to Editor" button
 * - Escape key to exit test mode
 * - Visual indicator when in test mode
 */

import React, { useEffect, useState } from 'react';

export interface TestOverlayProps {
  stateName: string;
  stateValue: number;
  onExit: () => void;
}

/**
 * TestOverlay Component
 *
 * Overlay shown during state testing mode.
 */
const TestOverlay: React.FC<TestOverlayProps> = ({
  stateName,
  stateValue,
  onExit,
}) => {
  const [countdown, setCountdown] = useState(5);

  /**
   * Handle keyboard events
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onExit();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onExit]);

  /**
   * Countdown to auto-hide
   */
  useEffect(() => {
    if (countdown <= 0) return;

    const timer = setTimeout(() => {
      setCountdown(countdown - 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [countdown]);

  /**
   * Format state name for display
   */
  const formatStateName = (name: string): string => {
    return name
      .split('_')
      .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(' ');
  };

  return (
    <div className="test-overlay">
      <div className="test-overlay-content">
        <div className="test-indicator">
          <span className="test-dot"></span>
          <span className="test-label">TEST MODE</span>
        </div>

        <div className="test-state-info">
          <div className="test-state-name">{formatStateName(stateName)}</div>
          <div className="test-state-value">State #{stateValue}</div>
        </div>

        <button className="test-exit-btn" onClick={onExit}>
          ← Return to Editor
        </button>

        {countdown > 0 && (
          <div className="test-countdown">
            Auto-hide in {countdown}s...
          </div>
        )}

        <div className="test-hint">
          Press <kbd>Escape</kbd> to exit
        </div>
      </div>
    </div>
  );
};

export default TestOverlay;
