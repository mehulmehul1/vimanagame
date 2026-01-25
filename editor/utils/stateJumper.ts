/**
 * stateJumper.ts - State Jump Utility for Scene Flow Navigator
 * =============================================================================
 *
 * Provides functionality to jump to a specific game state during testing.
 * Features:
 * - Jump to state by name and value
 * - URL parameter manipulation
 * - postMessage API for iframe communication
 * - Visual feedback during transition
 */

import type { PlayerPosition } from '../types/story';

/**
 * Jump to a specific game state
 * Updates URL parameters and optionally communicates with the game iframe
 *
 * @param stateName - The name of the state to jump to (e.g., "PHONE_BOOTH_RINGING")
 * @param stateValue - The numeric value of the state
 * @param options - Optional parameters for the jump
 */
export async function jumpToState(
  stateName: string,
  stateValue: number,
  options: {
    preserveViewport?: boolean;
    showFeedback?: boolean;
    playerPosition?: PlayerPosition;
  } = {}
): Promise<void> {
  const {
    preserveViewport = false,
    showFeedback = true,
    playerPosition,
  } = options;

  // Show loading feedback
  if (showFeedback) {
    showLoadingIndicator(stateName);
  }

  try {
    // Update URL with gameState parameter
    const url = new URL(window.location.href);
    url.searchParams.set('gameState', stateName);
    url.searchParams.set('gameStateValue', stateValue.toString());

    // Add player position if provided
    if (playerPosition) {
      url.searchParams.set('playerX', playerPosition.x.toString());
      url.searchParams.set('playerY', playerPosition.y.toString());
      url.searchParams.set('playerZ', playerPosition.z.toString());
      if (playerPosition.rotation) {
        url.searchParams.set('playerRotX', playerPosition.rotation.x.toString());
        url.searchParams.set('playerRotY', playerPosition.rotation.y.toString());
        url.searchParams.set('playerRotZ', playerPosition.rotation.z.toString());
      }
    }

    if (preserveViewport) {
      // Preserve viewport position
      const viewportParams = getViewportParams();
      viewportParams.forEach((value, key) => {
        url.searchParams.set(key, value);
      });
    }

    // Try to communicate with game iframe first
    const iframe = document.querySelector('iframe[src*="index.html"]');
    if (iframe && iframe.contentWindow) {
      // Send message to game iframe
      iframe.contentWindow.postMessage({
        type: 'JUMP_TO_STATE',
        stateName,
        stateValue,
        playerPosition,
      }, '*');

      // Wait a bit for the iframe to handle the message
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Reload the page with new state
    window.location.href = url.toString();

  } catch (error) {
    console.error('Failed to jump to state:', error);
    hideLoadingIndicator();
  }
}

/**
 * Get current viewport parameters for preservation
 */
function getViewportParams(): Map<string, string> {
  const params = new Map<string, string>();

  try {
    const camera = document.querySelector('[data-camera-position]');
    if (camera) {
      const position = camera.getAttribute('data-camera-position');
      if (position) {
        params.set('cameraX', position);
      }
    }
  } catch (error) {
    // Ignore viewport capture errors
  }

  return params;
}

/**
 * Show loading indicator during state transition
 */
function showLoadingIndicator(stateName: string): void {
  // Remove existing indicator if any
  hideLoadingIndicator();

  const indicator = document.createElement('div');
  indicator.id = 'state-jump-loading';
  indicator.className = 'state-jump-loading';
  indicator.innerHTML = `
    <div class="loading-content">
      <div class="loading-spinner"></div>
      <div class="loading-text">Jumping to ${formatStateName(stateName)}...</div>
    </div>
  `;

  document.body.appendChild(indicator);

  // Auto-hide after 5 seconds (fallback)
  setTimeout(() => {
    hideLoadingIndicator();
  }, 5000);
}

/**
 * Hide loading indicator
 */
function hideLoadingIndicator(): void {
  const indicator = document.getElementById('state-jump-loading');
  if (indicator && indicator.parentNode) {
    indicator.parentNode.removeChild(indicator);
  }
}

/**
 * Format state name for display
 */
function formatStateName(stateName: string): string {
  return stateName
    .split('_')
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(' ');
}

/**
 * Get current game state from URL parameters
 */
export function getCurrentState(): { name: string | null; value: number | null } {
  const params = new URLSearchParams(window.location.search);
  const name = params.get('gameState');
  const value = params.get('gameStateValue');

  return {
    name,
    value: value ? parseInt(value, 10) : null,
  };
}

/**
 * Clear game state from URL
 */
export function clearState(): void {
  const url = new URL(window.location.href);
  url.searchParams.delete('gameState');
  url.searchParams.delete('gameStateValue');
  url.searchParams.delete('playerX');
  url.searchParams.delete('playerY');
  url.searchParams.delete('playerZ');

  window.history.replaceState({}, '', url.toString());
}

/**
 * Check if currently in test mode
 */
export function isInTestMode(): boolean {
  return new URLSearchParams(window.location.search).has('testMode');
}

/**
 * Enable test mode
 */
export function enableTestMode(): void {
  const url = new URL(window.location.href);
  url.searchParams.set('testMode', 'true');
  window.history.replaceState({}, '', url.toString());
}

/**
 * Disable test mode
 */
export function disableTestMode(): void {
  const url = new URL(window.location.href);
  url.searchParams.delete('testMode');
  window.history.replaceState({}, '', url.toString());
}

export default {
  jumpToState,
  getCurrentState,
  clearState,
  isInTestMode,
  enableTestMode,
  disableTestMode,
};
