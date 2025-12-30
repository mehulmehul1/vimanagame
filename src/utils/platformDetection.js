/**
 * platformDetection.js - DEVICE AND BROWSER CAPABILITY DETECTION
 * =============================================================================
 *
 * ROLE: Detects platform capabilities (mobile, iOS, Safari, fullscreen support)
 * and sets corresponding flags in gameManager state.
 *
 * DETECTED CAPABILITIES:
 * - isMobile: Touch-capable device
 * - isIOS: iPhone/iPad (doesn't support Fullscreen API)
 * - isSafari: Safari browser (all versions)
 * - isFullscreenSupported: Fullscreen API availability
 *
 * USAGE:
 * Call detectPlatform(gameManager) early in initialization.
 * Other systems can then check gameState.isIOS, gameState.isMobile, etc.
 *
 * =============================================================================
 */

/**
 * Detect platform capabilities and set gameManager state
 * @param {GameManager} gameManager - GameManager instance to update state on
 */
export function detectPlatform(gameManager) {
  if (!gameManager) {
    console.warn("[PlatformDetection] gameManager not provided");
    return;
  }

  // Detect mobile/touch support
  const isMobile = "ontouchstart" in window || navigator.maxTouchPoints > 0;

  // Detect iOS (which doesn't support fullscreen API)
  const isIOS =
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1); // iPad with iOS 13+

  // Detect Safari browser (all versions, including macOS Safari)
  // Must explicitly check for Safari and exclude Chrome/Chromium/Edge
  const userAgent = navigator.userAgent.toLowerCase();
  const isSafari =
    (userAgent.includes("safari") &&
      !userAgent.includes("chrome") &&
      !userAgent.includes("chromium") &&
      !userAgent.includes("edge")) ||
    // Fallback: Check for Safari-specific features (but exclude Chrome which also has Apple vendor)
    (navigator.vendor &&
      navigator.vendor.indexOf("Apple") > -1 &&
      !userAgent.includes("chrome") &&
      !userAgent.includes("chromium"));

  // Check if any fullscreen API is available
  const isFullscreenSupported =
    !isIOS &&
    (document.fullscreenEnabled ||
      document.webkitFullscreenEnabled ||
      document.mozFullScreenEnabled ||
      document.msFullscreenEnabled);

  gameManager.setState({
    isMobile: isMobile,
    isIOS: isIOS,
    isSafari: isSafari,
    isFullscreenSupported: isFullscreenSupported,
  });

  return { isMobile, isIOS, isSafari, isFullscreenSupported };
}
