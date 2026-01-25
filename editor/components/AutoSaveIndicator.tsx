import React, { useState, useEffect } from 'react';
import AutoSaveManager from '../core/AutoSaveManager';
import './AutoSaveIndicator.css';

/**
 * AutoSaveIndicator - Show auto-save status and last save time
 *
 * Features:
 * - Show last save time
 * - Show time until next save
 * - Visual indicator when saving
 * - Manual save button
 */
interface AutoSaveIndicatorProps {
    autoSaveManager: AutoSaveManager;
    interval?: number; // Auto-save interval in minutes
}

const AutoSaveIndicator: React.FC<AutoSaveIndicatorProps> = ({ autoSaveManager, interval = 2 }) => {
    const [lastSaveTime, setLastSaveTime] = useState<number>(0);
    const [isSaving, setIsSaving] = useState(false);
    const [timeUntilNextSave, setTimeUntilNextSave] = useState<number>(0);

    useEffect(() => {
        // Enable auto-save
        autoSaveManager.enable(interval);

        // Listen for auto-save events
        const handleAutoSaved = (data: any) => {
            setLastSaveTime(data.timestamp);
            setIsSaving(true);
            setTimeout(() => setIsSaving(false), 1000);
        };

        autoSaveManager.on('autoSaved', handleAutoSaved);

        // Update time until next save every second
        const updateTimer = () => {
            setTimeUntilNextSave(autoSaveManager.getTimeUntilNextSave());
        };

        const timerInterval = setInterval(updateTimer, 1000);
        updateTimer();

        return () => {
            autoSaveManager.off('autoSaved', handleAutoSaved);
            clearInterval(timerInterval);
        };
    }, [autoSaveManager, interval]);

    /**
     * Format milliseconds to readable time
     */
    const formatTime = (ms: number): string => {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;

        if (minutes > 0) {
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            return `${remainingSeconds}s`;
        }
    };

    /**
     * Format timestamp to readable date
     */
    const formatTimestamp = (timestamp: number): string => {
        if (!timestamp) return 'Never';

        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 1) {
            return 'Just now';
        } else if (diffMins < 60) {
            return `${diffMins}m ago`;
        } else {
            const hours = Math.floor(diffMins / 60);
            return `${hours}h ago`;
        }
    };

    /**
     * Handle manual save
     */
    const handleManualSave = async () => {
        setIsSaving(true);
        await autoSaveManager.saveNow('manual');
        setTimeout(() => setIsSaving(false), 1000);
    };

    return (
        <div className={`autosave-indicator ${isSaving ? 'saving' : ''}`}>
            <div className="autosave-icon">
                {isSaving ? '💾' : '📁'}
            </div>
            <div className="autosave-text">
                <div className="autosave-label">Auto-save</div>
                <div className="autosave-time">
                    {lastSaveTime ? `Saved ${formatTimestamp(lastSaveTime)}` : 'Not saved yet'}
                </div>
                {timeUntilNextSave > 0 && (
                    <div className="autosave-next">
                        Next in {formatTime(timeUntilNextSave)}
                    </div>
                )}
            </div>
            <button
                className="autosave-save-button"
                onClick={handleManualSave}
                disabled={isSaving}
                title="Save now (Ctrl+S)"
            >
                {isSaving ? 'Saving...' : 'Save'}
            </button>
        </div>
    );
};

export default AutoSaveIndicator;
