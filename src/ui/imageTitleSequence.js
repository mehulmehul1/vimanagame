/**
 * imageTitleSequence.js - IMAGE-BASED TITLE SEQUENCE
 * =============================================================================
 *
 * ROLE: Lightweight fallback title sequence using image fade-in/fade-out.
 * Used when particle effects are too CPU-intensive.
 *
 * KEY RESPONSIBILITIES:
 * - Fade in text images with staggered timing
 * - Hold for specified duration
 * - Fade out all elements
 * - Fire onComplete callback
 *
 * =============================================================================
 */
export class ImageTitleSequence {
  constructor(texts, options = {}) {
    this.texts = texts;
    this.introDuration = options.introDuration || 3.0;
    this.staggerDelay = options.staggerDelay || 2.0;
    this.holdDuration = options.holdDuration || 3.0;
    this.outroDuration = options.outroDuration || 2.0;
    this.onComplete = options.onComplete || null;

    this.time = 0;
    this.completed = false;
    this.totalDuration =
      this.introDuration +
      this.staggerDelay * (texts.length - 1) +
      this.holdDuration +
      this.outroDuration;

    this.outroStartTime =
      this.introDuration +
      this.staggerDelay * (texts.length - 1) +
      this.holdDuration;

    // Initialize text timing
    this.texts.forEach((text, i) => {
      text._startTime = i * this.staggerDelay;
    });
  }

  update(dt) {
    // Early return if sequence is complete
    if (this.isComplete()) {
      if (!this.completed) {
        this.completed = true;
        if (this.onComplete) {
          this.onComplete();
        }
      }
      return;
    }

    this.time += dt;

    // Update each text element
    this.texts.forEach((text) => {
      if (!text.element) return;

      const localTime = this.time - text._startTime;
      const outroTime = this.time - this.outroStartTime;

      let opacity = 0;

      if (localTime < 0) {
        // Before intro
        opacity = 0;
      } else if (localTime < this.introDuration) {
        // Fade in
        opacity = localTime / this.introDuration;
      } else if (outroTime < 0) {
        // Hold phase
        opacity = 1.0;
      } else if (outroTime < this.outroDuration) {
        // Fade out
        opacity = 1.0 - outroTime / this.outroDuration;
      } else {
        // After outro
        opacity = 0;
      }

      text.element.style.opacity = Math.max(0, Math.min(1, opacity));
    });
  }

  isComplete() {
    return this.time >= this.totalDuration;
  }

  hasOutroStarted() {
    return this.time >= this.outroStartTime && this.time < this.totalDuration;
  }
}

