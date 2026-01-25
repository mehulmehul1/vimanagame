/**
 * ConfirmDialog.tsx - Confirmation Dialog Component
 * =============================================================================
 *
 * Reusable modal dialog for confirming destructive or important actions.
 * Features:
 * - Modal overlay with backdrop
 * - Keyboard handlers (Enter=Confirm, Escape=Cancel)
 * - Focus management (trap focus in modal)
 * - Customizable title and message
 */

import React, { useEffect, useRef } from 'react';

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * ConfirmDialog Component
 *
 * Modal dialog for confirming user actions.
 */
const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'danger',
  onConfirm,
  onCancel,
}) => {
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  /**
   * Handle keyboard events
   */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        onConfirm();
        break;
      case 'Escape':
        e.preventDefault();
        onCancel();
        break;
      case 'Tab':
        // Trap focus within modal
        if (!modalRef.current) return;
        const focusableElements = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
        break;
    }
  };

  /**
   * Focus confirm button on open
   */
  useEffect(() => {
    if (open) {
      setTimeout(() => {
        confirmButtonRef.current?.focus();
      }, 50);
    }
  }, [open]);

  /**
   * Prevent body scroll when modal is open
   */
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [open]);

  if (!open) return null;

  const variantClass = `confirm-dialog-${variant}`;

  return (
    <div
      className="confirm-dialog-overlay"
      onClick={onCancel}
      onKeyDown={handleKeyDown}
    >
      <div
        className={`confirm-dialog ${variantClass}`}
        ref={modalRef}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
        aria-describedby="dialog-message"
      >
        <div className="confirm-dialog-header">
          <h3 id="dialog-title" className="confirm-dialog-title">{title}</h3>
        </div>

        <div className="confirm-dialog-body">
          <p id="dialog-message" className="confirm-dialog-message">{message}</p>
        </div>

        <div className="confirm-dialog-footer">
          <button
            className="confirm-dialog-btn confirm-dialog-btn-cancel"
            onClick={onCancel}
            type="button"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmButtonRef}
            className={`confirm-dialog-btn confirm-dialog-btn-confirm confirm-dialog-btn-confirm-${variant}`}
            onClick={onConfirm}
            type="button"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
