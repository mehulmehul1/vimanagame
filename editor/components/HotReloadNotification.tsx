import React, { useState, useEffect } from 'react';
import DataManager from '../core/DataManager';
import './HotReloadNotification.css';

/**
 * HotReloadNotification - Show notification when scene file is modified externally
 *
 * Features:
 * - Notification when sceneData.js is modified
 * - Reload button to apply changes
 * - Auto-hide after timeout
 * - Optional auto-reload
 */
interface HotReloadNotificationProps {
    dataManager: DataManager;
    onReload?: () => void;
}

const HotReloadNotification: React.FC<HotReloadNotificationProps> = ({ dataManager, onReload }) => {
    const [showNotification, setShowNotification] = useState(false);
    const [fileName, setFileName] = useState('');

    useEffect(() => {
        // Listen for external file changes
        const handleExternalFileChange = (data: any) => {
            const file = data.file || 'sceneData.js';
            setFileName(file);
            setShowNotification(true);

            // Auto-hide after 10 seconds
            setTimeout(() => {
                setShowNotification(false);
            }, 10000);
        };

        // Listen for hot reload complete
        const handleHotReloadComplete = () => {
            setShowNotification(false);
        };

        dataManager.on('externalFileChange', handleExternalFileChange);
        dataManager.on('hotReloadComplete', handleHotReloadComplete);

        return () => {
            dataManager.off('externalFileChange', handleExternalFileChange);
            dataManager.off('hotReloadComplete', handleHotReloadComplete);
        };
    }, [dataManager]);

    /**
     * Handle reload button click
     */
    const handleReload = async () => {
        try {
            await dataManager.reloadScene();
            setShowNotification(false);
            if (onReload) {
                onReload();
            }
        } catch (error) {
            console.error('HotReloadNotification: Failed to reload:', error);
        }
    };

    /**
     * Handle dismiss button click
     */
    const handleDismiss = () => {
        setShowNotification(false);
    };

    if (!showNotification) {
        return null;
    }

    return (
        <div className="hot-reload-notification">
            <div className="notification-content">
                <div className="notification-icon">📄</div>
                <div className="notification-text">
                    <div className="notification-title">Scene Modified Externally</div>
                    <div className="notification-message">
                        {fileName} has been modified. Reload to apply changes?
                    </div>
                </div>
                <div className="notification-actions">
                    <button
                        className="notification-button primary"
                        onClick={handleReload}
                        title="Reload scene"
                    >
                        🔄 Reload
                    </button>
                    <button
                        className="notification-button secondary"
                        onClick={handleDismiss}
                        title="Dismiss notification"
                    >
                        ✕ Dismiss
                    </button>
                </div>
            </div>
        </div>
    );
};

export default HotReloadNotification;
