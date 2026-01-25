import React, { useState, useEffect, useRef } from 'react';
import AssetImporter from '../../utils/AssetImporter';
import './AssetBrowser.css';

/**
 * Asset file information
 */
interface AssetFile {
    name: string;
    path: string;
    type: 'model' | 'image' | 'audio' | 'unknown';
    extension: string;
    size?: number;
    thumbnail?: string;
}

/**
 * Asset Browser Panel - Asset management and preview
 *
 * Features:
 * - Scan assets/ directory recursively
 * - Grid view with thumbnails
 * - Folder navigation breadcrumb
 * - Asset preview panel on selection
 * - Search and filter
 * - File type icons (.glb, .gltf, .sog, .ply, .png, .jpg, .mp3, .wav)
 */
interface AssetBrowserProps {
    assetsPath?: string;
}

const AssetBrowser: React.FC<AssetBrowserProps> = ({ assetsPath = 'assets' }) => {
    const [assets, setAssets] = useState<AssetFile[]>([]);
    const [selectedAsset, setSelectedAsset] = useState<AssetFile | null>(null);
    const [currentPath, setCurrentPath] = useState<string>(assetsPath);
    const [searchQuery, setSearchQuery] = useState<string>('');
    const [filterType, setFilterType] = useState<string>('all');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [isDragging, setIsDragging] = useState<boolean>(false);
    const assetImporterRef = useRef<AssetImporter | null>(null);

    useEffect(() => {
        loadAssets();
        assetImporterRef.current = AssetImporter.getInstance();
        assetImporterRef.current.setAssetsPath(assetsPath);
    }, [currentPath, assetsPath]);

    /**
     * Load assets from directory
     */
    const loadAssets = async () => {
        setIsLoading(true);
        try {
            // In a real implementation, this would scan the file system
            // For now, we'll use placeholder assets
            const placeholderAssets: AssetFile[] = [
                {
                    name: 'sample_box.glb',
                    path: `${currentPath}/sample_box.glb`,
                    type: 'model',
                    extension: 'glb',
                    size: 1024000
                },
                {
                    name: 'character.gltf',
                    path: `${currentPath}/character.gltf`,
                    type: 'model',
                    extension: 'gltf',
                    size: 2048000
                },
                {
                    name: 'texture.png',
                    path: `${currentPath}/texture.png`,
                    type: 'image',
                    extension: 'png',
                    size: 512000
                },
                {
                    name: 'background.jpg',
                    path: `${currentPath}/background.jpg`,
                    type: 'image',
                    extension: 'jpg',
                    size: 1024000
                },
                {
                    name: 'ambient.mp3',
                    path: `${currentPath}/ambient.mp3`,
                    type: 'audio',
                    extension: 'mp3',
                    size: 3072000
                },
                {
                    name: 'effect.wav',
                    path: `${currentPath}/effect.wav`,
                    type: 'audio',
                    extension: 'wav',
                    size: 512000
                }
            ];

            setAssets(placeholderAssets);
        } catch (error) {
            console.error('AssetBrowser: Failed to load assets', error);
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * Get icon for file type
     */
    const getFileIcon = (asset: AssetFile): string => {
        switch (asset.type) {
            case 'model':
                return '📦';
            case 'image':
                return '🖼️';
            case 'audio':
                return '🎵';
            default:
                return '📄';
        }
    };

    /**
     * Format file size
     */
    const formatFileSize = (bytes?: number): string => {
        if (!bytes) return 'Unknown';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    /**
     * Filter assets based on search and type
     */
    const filteredAssets = assets.filter(asset => {
        const matchesSearch = asset.name.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesType = filterType === 'all' || asset.type === filterType;
        return matchesSearch && matchesType;
    });

    /**
     * Handle asset selection
     */
    const handleAssetClick = (asset: AssetFile) => {
        setSelectedAsset(asset);
    };

    /**
     * Handle asset double-click (could open/preview)
     */
    const handleAssetDoubleClick = (asset: AssetFile) => {
        console.log('AssetBrowser: Double-clicked asset', asset);
        // TODO: Implement asset preview/open
    };

    /**
     * Get breadcrumb parts
     */
    const getBreadcrumbParts = (): string[] => {
        return currentPath.split('/').filter(part => part);
    };

    /**
     * Handle import button click
     */
    const handleImportClick = async () => {
        if (!assetImporterRef.current) return;

        try {
            const importedAssets = await assetImporterRef.current.openFilePicker({
                generateThumbnail: true,
                maxSize: 50 * 1024 * 1024 // 50MB
            });

            if (importedAssets.length > 0) {
                // Add to assets list
                const newAssets: AssetFile[] = importedAssets.map(asset => ({
                    name: asset.name,
                    path: asset.path,
                    type: asset.type,
                    extension: asset.extension,
                    size: asset.size,
                    thumbnail: asset.thumbnail
                }));

                setAssets(prev => [...prev, ...newAssets]);
                console.log(`AssetBrowser: Imported ${newAssets.length} assets`);
            }
        } catch (error) {
            console.error('AssetBrowser: Import failed', error);
        }
    };

    /**
     * Handle drag over
     */
    const handleDragOver = (event: React.DragEvent) => {
        event.preventDefault();
        setIsDragging(true);
    };

    /**
     * Handle drag leave
     */
    const handleDragLeave = (event: React.DragEvent) => {
        event.preventDefault();
        setIsDragging(false);
    };

    /**
     * Handle drop
     */
    const handleDrop = async (event: React.DragEvent) => {
        event.preventDefault();
        setIsDragging(false);

        if (!assetImporterRef.current) return;

        const files = Array.from(event.dataTransfer.files);
        if (files.length === 0) return;

        try {
            const importedAssets = await assetImporterRef.current.importFiles(files, {
                generateThumbnail: true,
                maxSize: 50 * 1024 * 1024
            });

            if (importedAssets.length > 0) {
                const newAssets: AssetFile[] = importedAssets.map(asset => ({
                    name: asset.name,
                    path: asset.path,
                    type: asset.type,
                    extension: asset.extension,
                    size: asset.size,
                    thumbnail: asset.thumbnail
                }));

                setAssets(prev => [...prev, ...newAssets]);
                console.log(`AssetBrowser: Dropped and imported ${newAssets.length} assets`);
            }
        } catch (error) {
            console.error('AssetBrowser: Drop import failed', error);
        }
    };

    return (
        <div
            className={`asset-browser-container ${isDragging ? 'drag-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Header */}
            <div className="asset-browser-header">
                <h3>Asset Browser</h3>
                <div className="header-actions">
                    <button
                        className="import-button"
                        onClick={handleImportClick}
                        title="Import Assets"
                    >
                        📥 Import
                    </button>
                    <button
                        className="refresh-button"
                        onClick={loadAssets}
                        title="Refresh Assets"
                    >
                        🔄
                    </button>
                </div>
            </div>

            {/* Breadcrumb */}
            <div className="asset-browser-breadcrumb">
                {getBreadcrumbParts().map((part, index, array) => (
                    <React.Fragment key={index}>
                        <span
                            className="breadcrumb-part"
                            onClick={() => {
                                const newPath = array.slice(0, index + 1).join('/');
                                setCurrentPath(newPath);
                            }}
                        >
                            {part}
                        </span>
                        {index < array.length - 1 && <span className="breadcrumb-separator">/</span>}
                    </React.Fragment>
                ))}
            </div>

            {/* Search and Filter */}
            <div className="asset-browser-controls">
                <input
                    type="text"
                    className="search-input"
                    placeholder="Search assets..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
                <select
                    className="filter-select"
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                >
                    <option value="all">All Types</option>
                    <option value="model">Models</option>
                    <option value="image">Images</option>
                    <option value="audio">Audio</option>
                </select>
            </div>

            {/* Content */}
            <div className="asset-browser-content">
                {/* Asset Grid */}
                <div className="asset-grid">
                    {isLoading ? (
                        <div className="asset-loading">
                            <div className="loading-spinner" />
                            <p>Loading assets...</p>
                        </div>
                    ) : filteredAssets.length === 0 ? (
                        <div className="asset-empty">
                            <p>No assets found</p>
                            <p className="hint">Import files or check the assets directory</p>
                        </div>
                    ) : (
                        filteredAssets.map((asset) => (
                            <div
                                key={asset.path}
                                className={`asset-item ${selectedAsset === asset ? 'selected' : ''}`}
                                onClick={() => handleAssetClick(asset)}
                                onDoubleClick={() => handleAssetDoubleClick(asset)}
                                title={asset.name}
                            >
                                {asset.type === 'image' ? (
                                    <div className="asset-thumbnail">
                                        <img src={asset.path} alt={asset.name} onError={(e) => {
                                            (e.target as HTMLImageElement).style.display = 'none';
                                        }} />
                                    </div>
                                ) : (
                                    <div className="asset-icon">{getFileIcon(asset)}</div>
                                )}
                                <div className="asset-name">{asset.name}</div>
                                <div className="asset-size">{formatFileSize(asset.size)}</div>
                            </div>
                        ))
                    )}
                </div>

                {/* Preview Panel */}
                {selectedAsset && (
                    <div className="asset-preview">
                        <div className="preview-header">
                            <h4>Preview</h4>
                            <button
                                className="close-preview"
                                onClick={() => setSelectedAsset(null)}
                            >
                                ×
                            </button>
                        </div>
                        <div className="preview-content">
                            {selectedAsset.type === 'image' ? (
                                <img
                                    src={selectedAsset.path}
                                    alt={selectedAsset.name}
                                    className="preview-image"
                                />
                            ) : (
                                <div className="preview-placeholder">
                                    <div className="preview-icon">{getFileIcon(selectedAsset)}</div>
                                    <p>{selectedAsset.type}</p>
                                </div>
                            )}
                            <div className="preview-info">
                                <div className="info-row">
                                    <span className="info-label">Name:</span>
                                    <span className="info-value">{selectedAsset.name}</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Type:</span>
                                    <span className="info-value">{selectedAsset.type}</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Size:</span>
                                    <span className="info-value">{formatFileSize(selectedAsset.size)}</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Path:</span>
                                    <span className="info-value">{selectedAsset.path}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="asset-browser-footer">
                <small className="hint">
                    {filteredAssets.length} assets • Double-click to preview • Drag to viewport
                </small>
            </div>
        </div>
    );
};

export default AssetBrowser;
