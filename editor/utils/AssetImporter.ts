/**
 * AssetImporter - File import pipeline for the Shadow Web Editor
 *
 * Features:
 * - File picker dialog (input type="file")
 * - Copy file to assets directory
 * - Generate thumbnail for images/models
 * - Add to Asset Browser
 * - Support drag-drop into viewport
 * - Validation (file types, sizes)
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */

interface AssetImportOptions {
    maxSize?: number; // Maximum file size in bytes (default: 50MB)
    allowedTypes?: string[]; // Allowed file extensions
    generateThumbnail?: boolean; // Whether to generate thumbnails
    onProgress?: (progress: number) => void; // Progress callback
}

interface ImportedAsset {
    file: File;
    name: string;
    path: string;
    type: 'model' | 'image' | 'audio' | 'unknown';
    extension: string;
    size: number;
    thumbnail?: string;
    url: string;
}

class AssetImporter {
    private static instance: AssetImporter;
    private assetsPath: string = 'assets';
    private maxFileSize: number = 50 * 1024 * 1024; // 50MB default

    // Allowed file types
    private allowedModelTypes = ['glb', 'gltf', 'sog', 'ply', 'obj', 'fbx'];
    private allowedImageTypes = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'];
    private allowedAudioTypes = ['mp3', 'wav', 'ogg', 'aac', 'flac'];

    private constructor() {
        console.log('AssetImporter: Initialized');
    }

    public static getInstance(): AssetImporter {
        if (!AssetImporter.instance) {
            AssetImporter.instance = new AssetImporter();
        }
        return AssetImporter.instance;
    }

    /**
     * Set the assets directory path
     */
    public setAssetsPath(path: string): void {
        this.assetsPath = path;
    }

    /**
     * Get file type from extension
     */
    private getFileType(extension: string): ImportedAsset['type'] {
        const ext = extension.toLowerCase();

        if (this.allowedModelTypes.includes(ext)) return 'model';
        if (this.allowedImageTypes.includes(ext)) return 'image';
        if (this.allowedAudioTypes.includes(ext)) return 'audio';
        return 'unknown';
    }

    /**
     * Validate file type
     */
    private validateFileType(file: File): boolean {
        const extension = file.name.split('.').pop()?.toLowerCase() || '';
        const fileType = this.getFileType(extension);
        return fileType !== 'unknown';
    }

    /**
     * Validate file size
     */
    private validateFileSize(file: File, maxSize?: number): boolean {
        const limit = maxSize || this.maxFileSize;
        return file.size <= limit;
    }

    /**
     * Open file picker dialog
     */
    public async openFilePicker(options?: AssetImportOptions): Promise<ImportedAsset[]> {
        return new Promise((resolve, reject) => {
            const input = document.createElement('input');
            input.type = 'file';
            input.multiple = true;
            input.accept = this.buildAcceptString(options?.allowedTypes);

            input.onchange = async (event) => {
                const files = (event.target as HTMLInputElement).files;
                if (!files || files.length === 0) {
                    resolve([]);
                    return;
                }

                try {
                    const importedAssets = await this.importFiles(Array.from(files), options);
                    resolve(importedAssets);
                } catch (error) {
                    reject(error);
                }
            };

            input.oncancel = () => {
                resolve([]);
            };

            input.click();
        });
    }

    /**
     * Build accept string for file picker
     */
    private buildAcceptString(allowedTypes?: string[]): string {
        if (allowedTypes && allowedTypes.length > 0) {
            return '.' + allowedTypes.join(',.');
        }

        const modelAccept = this.allowedModelTypes.map(ext => `.${ext}`).join(',');
        const imageAccept = this.allowedImageTypes.map(ext => `.${ext}`).join(',');
        const audioAccept = this.allowedAudioTypes.map(ext => `.${ext}`).join(',');

        return `${modelAccept},${imageAccept},${audioAccept}`;
    }

    /**
     * Import files
     */
    public async importFiles(files: File[], options?: AssetImportOptions): Promise<ImportedAsset[]> {
        const importedAssets: ImportedAsset[] = [];
        const maxSize = options?.maxSize || this.maxFileSize;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (!file) continue;

            // Validate file type
            if (!this.validateFileType(file)) {
                console.warn(`AssetImporter: Invalid file type: ${file.name}`);
                continue;
            }

            // Validate file size
            if (!this.validateFileSize(file, maxSize)) {
                console.warn(`AssetImporter: File too large: ${file.name}`);
                continue;
            }

            try {
                const asset = await this.importFile(file, options);
                importedAssets.push(asset);

                // Report progress
                if (options?.onProgress) {
                    const progress = ((i + 1) / files.length) * 100;
                    options.onProgress(progress);
                }
            } catch (error) {
                console.error(`AssetImporter: Failed to import file: ${file.name}`, error);
            }
        }

        console.log(`AssetImporter: Imported ${importedAssets.length} files`);
        return importedAssets;
    }

    /**
     * Import a single file
     */
    private async importFile(file: File, options?: AssetImportOptions): Promise<ImportedAsset> {
        const extension = file.name.split('.').pop()?.toLowerCase() || '';
        const type = this.getFileType(extension);
        const path = `${this.assetsPath}/${file.name}`;

        // Create object URL for the file
        const url = URL.createObjectURL(file);

        const asset: ImportedAsset = {
            file,
            name: file.name,
            path,
            type,
            extension,
            size: file.size,
            url
        };

        // Generate thumbnail if requested
        if (options?.generateThumbnail !== false && type === 'image') {
            asset.thumbnail = await this.generateThumbnail(file);
        }

        return asset;
    }

    /**
     * Generate thumbnail for image
     */
    private async generateThumbnail(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    if (!ctx) {
                        reject(new Error('Failed to get canvas context'));
                        return;
                    }

                    // Calculate thumbnail size (max 128x128)
                    const maxSize = 128;
                    let width = img.width;
                    let height = img.height;

                    if (width > height) {
                        if (width > maxSize) {
                            height = (height * maxSize) / width;
                            width = maxSize;
                        }
                    } else {
                        if (height > maxSize) {
                            width = (width * maxSize) / height;
                            height = maxSize;
                        }
                    }

                    canvas.width = width;
                    canvas.height = height;

                    // Draw and export
                    ctx.drawImage(img, 0, 0, width, height);
                    const thumbnail = canvas.toDataURL('image/jpeg', 0.7);
                    resolve(thumbnail);
                };

                img.onerror = () => reject(new Error('Failed to load image'));
                img.src = event.target?.result as string;
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    /**
     * Setup drag and drop handlers for a container
     */
    public setupDragDrop(
        container: HTMLElement,
        onDrop: (assets: ImportedAsset[]) => void,
        options?: AssetImportOptions
    ): void {
        container.addEventListener('dragover', (event) => {
            event.preventDefault();
            event.stopPropagation();
            container.classList.add('drag-over');
        });

        container.addEventListener('dragleave', (event) => {
            event.preventDefault();
            event.stopPropagation();
            container.classList.remove('drag-over');
        });

        container.addEventListener('drop', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            container.classList.remove('drag-over');

            const dataTransfer = event.dataTransfer;
            if (!dataTransfer || !dataTransfer.files) return;

            const files = Array.from(dataTransfer.files);
            if (files.length === 0) return;

            try {
                const importedAssets = await this.importFiles(files, options);
                if (importedAssets.length > 0) {
                    onDrop(importedAssets);
                }
            } catch (error) {
                console.error('AssetImporter: Failed to import dropped files', error);
            }
        });
    }

    /**
     * Get file size as human-readable string
     */
    public formatFileSize(bytes: number): string {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    }

    /**
     * Cleanup object URLs
     */
    public revokeObjectURL(url: string): void {
        URL.revokeObjectURL(url);
    }

    /**
     * Validate and get file info
     */
    public getFileInfo(file: File): {
        valid: boolean;
        type: ImportedAsset['type'];
        extension: string;
        size: number;
        sizeFormatted: string;
        error?: string;
    } {
        const extension = file.name.split('.').pop()?.toLowerCase() || '';
        const type = this.getFileType(extension);

        if (type === 'unknown') {
            return {
                valid: false,
                type,
                extension,
                size: file.size,
                sizeFormatted: this.formatFileSize(file.size),
                error: 'Invalid file type'
            };
        }

        if (!this.validateFileSize(file)) {
            return {
                valid: false,
                type,
                extension,
                size: file.size,
                sizeFormatted: this.formatFileSize(file.size),
                error: 'File too large'
            };
        }

        return {
            valid: true,
            type,
            extension,
            size: file.size,
            sizeFormatted: this.formatFileSize(file.size)
        };
    }
}

export default AssetImporter;
