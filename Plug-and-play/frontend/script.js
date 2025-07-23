/**
 * Frontend JavaScript for LeRobot Installation Assistant
 * Handles UI interactions and real-time communication with backend
 */

class InstallationApp {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.installationRunning = false;
        
        this.initializeElements();
        this.bindEvents();
        this.connectSocket();
        this.checkSystemInfo();
    }
    
    /**
     * Initialize DOM element references
     */
    initializeElements() {
        // Form elements
        this.installationPath = document.getElementById('installationPath');
        this.browseBtn = document.getElementById('browseBtn');
        this.startInstallBtn = document.getElementById('startInstallBtn');
        this.cancelInstallBtn = document.getElementById('cancelInstallBtn');
        
        // Progress elements
        this.progressFill = document.getElementById('progressFill');
        this.progressPercent = document.getElementById('progressPercent');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusIcon = document.getElementById('statusIcon');
        this.statusText = document.getElementById('statusText');
        this.stepsContainer = document.getElementById('stepsContainer');
        
        // System status elements
        this.gitIcon = document.getElementById('gitIcon');
        this.gitStatus = document.getElementById('gitStatus');
        this.condaIcon = document.getElementById('condaIcon');
        this.condaStatus = document.getElementById('condaStatus');
        
        // Log elements
        this.logContainer = document.getElementById('logContainer');
        this.clearLogBtn = document.getElementById('clearLogBtn');
        
        // Modal elements
        this.successModal = document.getElementById('successModal');
        this.errorModal = document.getElementById('errorModal');
        this.closeSuccessModal = document.getElementById('closeSuccessModal');
        this.closeErrorModal = document.getElementById('closeErrorModal');
        this.errorMessage = document.getElementById('errorMessage');
    }
    
    /**
     * Bind event handlers
     */
    bindEvents() {
        this.browseBtn.addEventListener('click', () => this.browseDirectory());
        this.startInstallBtn.addEventListener('click', () => this.startInstallation());
        this.cancelInstallBtn.addEventListener('click', () => this.cancelInstallation());
        this.clearLogBtn.addEventListener('click', () => this.clearLog());
        this.closeSuccessModal.addEventListener('click', () => this.hideSuccessModal());
        this.closeErrorModal.addEventListener('click', () => this.hideErrorModal());
        
        // Close modals when clicking outside
        this.successModal.addEventListener('click', (e) => {
            if (e.target === this.successModal) this.hideSuccessModal();
        });
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) this.hideErrorModal();
        });
        
        // Handle escape key for modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideSuccessModal();
                this.hideErrorModal();
            }
        });
    }
    
    /**
     * Connect to WebSocket server
     */
    connectSocket() {
        try {
            this.socket = io('http://127.0.0.1:5000');
            
            this.socket.on('connect', () => {
                this.isConnected = true;
                this.addLogMessage('Connected to installation service', 'info');
            });
            
            this.socket.on('disconnect', () => {
                this.isConnected = false;
                this.addLogMessage('Disconnected from installation service', 'warning');
            });
            
            this.socket.on('log_message', (data) => {
                this.handleLogMessage(data.message);
            });
            
            this.socket.on('progress_update', (data) => {
                this.handleProgressUpdate(data);
            });
            
            this.socket.on('error_message', (data) => {
                this.handleErrorMessage(data.message);
            });
            
            this.socket.on('installation_complete', (data) => {
                this.handleInstallationComplete(data.message);
            });
            
            this.socket.on('start_port_detection', (data) => {
                this.handlePortDetectionStart(data);
            });
            
            this.socket.on('status_update', (data) => {
                this.handleStatusUpdate(data);
            });
            
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.addLogMessage('Failed to connect to installation service', 'error');
        }
    }
    
    /**
     * Check system information and prerequisites
     */
    async checkSystemInfo() {
        try {
            const response = await fetch('/api/system-info');
            const data = await response.json();
            
            this.updateSystemStatus('git', data.git_available);
            this.updateSystemStatus('conda', data.conda_available);
            
            if (data.default_path && !this.installationPath.value) {
                this.installationPath.value = data.default_path;
            }
            
        } catch (error) {
            console.error('Failed to check system info:', error);
            this.addLogMessage('Failed to check system prerequisites', 'error');
        }
    }
    
    /**
     * Update system status indicators
     */
    updateSystemStatus(tool, available) {
        const icon = tool === 'git' ? this.gitIcon : this.condaIcon;
        const status = tool === 'git' ? this.gitStatus : this.condaStatus;
        
        icon.className = 'fas fa-' + (tool === 'git' ? 'code-branch' : 'cube');
        
        if (available) {
            icon.classList.add('status-available');
            status.textContent = tool.charAt(0).toUpperCase() + tool.slice(1) + ' Available';
        } else {
            icon.classList.add('status-unavailable');
            status.textContent = tool.charAt(0).toUpperCase() + tool.slice(1) + ' Not Found';
        }
    }
    
    /**
     * Browse for installation directory
     */
    async browseDirectory() {
        this.showDirectoryBrowser();
    }
    
    /**
     * Show directory browser modal
     */
    async showDirectoryBrowser() {
        // Create directory browser modal
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.id = 'directoryBrowserModal';
        
        modal.innerHTML = `
            <div class="modal-content directory-browser">
                <div class="modal-header">
                    <i class="fas fa-folder-open"></i>
                    <h3>Select Installation Directory</h3>
                    <button type="button" class="modal-close" id="closeBrowserModal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="browser-toolbar">
                        <button type="button" class="btn btn-ghost btn-small" id="browserBack" disabled>
                            <i class="fas fa-arrow-left"></i>
                            Back
                        </button>
                        <button type="button" class="btn btn-ghost btn-small" id="browserHome">
                            <i class="fas fa-home"></i>
                            Home
                        </button>
                        <div class="path-display">
                            <input type="text" id="currentPath" class="path-input" readonly>
                        </div>
                        <button type="button" class="btn btn-ghost btn-small" id="createFolder">
                            <i class="fas fa-folder-plus"></i>
                            New Folder
                        </button>
                    </div>
                    <div class="browser-content">
                        <div class="browser-loading" id="browserLoading">
                            <i class="fas fa-spinner fa-spin"></i>
                            Loading...
                        </div>
                        <div class="file-list" id="fileList"></div>
                    </div>
                    <div class="quick-access">
                        <h4>Quick Access</h4>
                        <div class="quick-buttons" id="quickButtons"></div>
                    </div>
                </div>
                <div class="modal-footer">
                    <div class="selected-path">
                        <span class="selected-label">Selected:</span>
                        <span class="selected-value" id="selectedPath"></span>
                    </div>
                    <div class="modal-buttons">
                        <button type="button" class="btn btn-secondary" id="cancelBrowser">
                            <i class="fas fa-times"></i>
                            Cancel
                        </button>
                        <button type="button" class="btn btn-primary" id="selectDirectory" disabled>
                            <i class="fas fa-check"></i>
                            Select This Folder
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Initialize directory browser
        this.initDirectoryBrowser(modal);
    }
    
    /**
     * Initialize directory browser functionality
     */
    async initDirectoryBrowser(modal) {
        this.currentBrowserPath = '';
        this.browserHistory = [];
        
        // Get DOM elements
        const backBtn = modal.querySelector('#browserBack');
        const homeBtn = modal.querySelector('#browserHome');
        const currentPathInput = modal.querySelector('#currentPath');
        const createFolderBtn = modal.querySelector('#createFolder');
        const browserLoading = modal.querySelector('#browserLoading');
        const fileList = modal.querySelector('#fileList');
        const quickButtons = modal.querySelector('#quickButtons');
        const selectedPathSpan = modal.querySelector('#selectedPath');
        const selectBtn = modal.querySelector('#selectDirectory');
        const cancelBtn = modal.querySelector('#cancelBrowser');
        const closeBtn = modal.querySelector('#closeBrowserModal');
        
        // Event handlers
        backBtn.addEventListener('click', () => this.browserGoBack());
        homeBtn.addEventListener('click', () => this.browserGoHome());
        createFolderBtn.addEventListener('click', () => this.createNewFolder());
        selectBtn.addEventListener('click', () => this.selectCurrentDirectory(modal));
        cancelBtn.addEventListener('click', () => this.closeBrowserModal(modal));
        closeBtn.addEventListener('click', () => this.closeBrowserModal(modal));
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeBrowserModal(modal);
        });
        
        // Load initial directory and quick access
        await this.loadQuickAccess();
        await this.loadDirectory();
    }
    
    /**
     * Load quick access buttons
     */
    async loadQuickAccess() {
        try {
            const response = await fetch('/api/get-home-directory');
            const data = await response.json();
            
            if (data.suggested_paths) {
                const quickButtons = document.querySelector('#quickButtons');
                quickButtons.innerHTML = '';
                
                data.suggested_paths.forEach(path => {
                    const pathObj = path.split('/').pop() || path.split('\\').pop() || path;
                    const button = document.createElement('button');
                    button.className = 'btn btn-ghost btn-small quick-button';
                    button.innerHTML = `<i class="fas fa-folder"></i> ${pathObj}`;
                    button.onclick = () => this.loadDirectory(path);
                    quickButtons.appendChild(button);
                });
            }
        } catch (error) {
            console.error('Failed to load quick access:', error);
        }
    }
    
    /**
     * Load directory contents
     */
    async loadDirectory(path = null) {
        const browserLoading = document.querySelector('#browserLoading');
        const fileList = document.querySelector('#fileList');
        const currentPathInput = document.querySelector('#currentPath');
        const backBtn = document.querySelector('#browserBack');
        const selectedPathSpan = document.querySelector('#selectedPath');
        const selectBtn = document.querySelector('#selectDirectory');
        
        // Show loading state
        browserLoading.style.display = 'flex';
        fileList.style.display = 'none';
        
        try {
            const response = await fetch('/api/browse-directory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path || this.currentBrowserPath })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update state
            this.currentBrowserPath = data.current_path;
            currentPathInput.value = this.currentBrowserPath;
            selectedPathSpan.textContent = this.currentBrowserPath;
            
            // Enable/disable back button
            backBtn.disabled = !data.items.some(item => item.type === 'parent');
            
            // Enable select button
            selectBtn.disabled = false;
            
            // Render file list
            this.renderFileList(data.items);
            
        } catch (error) {
            console.error('Failed to load directory:', error);
            this.showBrowserError(error.message);
        } finally {
            browserLoading.style.display = 'none';
            fileList.style.display = 'block';
        }
    }
    
    /**
     * Render file list
     */
    renderFileList(items) {
        const fileList = document.querySelector('#fileList');
        fileList.innerHTML = '';
        
        items.forEach(item => {
            const fileItem = document.createElement('div');
            fileItem.className = `file-item ${item.type}`;
            
            const icon = item.type === 'parent' ? 'level-up-alt' :
                        item.is_dir ? 'folder' : 'file';
            
            fileItem.innerHTML = `
                <div class="file-icon">
                    <i class="fas fa-${icon}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${item.name}</div>
                    <div class="file-meta">
                        ${item.type === 'directory' ? 'Folder' : 
                          item.type === 'parent' ? 'Parent Directory' : 'File'}
                    </div>
                </div>
            `;
            
            // Add click handler for directories
            if (item.is_dir) {
                fileItem.style.cursor = 'pointer';
                fileItem.addEventListener('click', () => {
                    if (item.type === 'parent') {
                        this.browserGoBack();
                    } else {
                        this.loadDirectory(item.path);
                    }
                });
                
                fileItem.addEventListener('dblclick', () => {
                    if (item.type !== 'parent') {
                        this.loadDirectory(item.path);
                    }
                });
            }
            
            fileList.appendChild(fileItem);
        });
    }
    
    /**
     * Go back to parent directory
     */
    browserGoBack() {
        const currentPath = this.currentBrowserPath;
        const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
        this.loadDirectory(parentPath);
    }
    
    /**
     * Go to home directory
     */
    async browserGoHome() {
        try {
            const response = await fetch('/api/get-home-directory');
            const data = await response.json();
            if (data.home_path) {
                this.loadDirectory(data.home_path);
            }
        } catch (error) {
            console.error('Failed to go home:', error);
        }
    }
    
    /**
     * Create new folder
     */
    async createNewFolder() {
        const folderName = prompt('Enter folder name:');
        if (!folderName || !folderName.trim()) return;
        
        try {
            const response = await fetch('/api/create-directory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    parent_path: this.currentBrowserPath,
                    directory_name: folderName.trim()
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                alert(`Failed to create folder: ${data.error}`);
            } else {
                // Reload current directory to show new folder
                await this.loadDirectory(this.currentBrowserPath);
            }
        } catch (error) {
            console.error('Failed to create folder:', error);
            alert('Failed to create folder');
        }
    }
    
    /**
     * Select current directory and close browser
     */
    selectCurrentDirectory(modal) {
        this.installationPath.value = this.currentBrowserPath;
        this.closeBrowserModal(modal);
    }
    
    /**
     * Close directory browser modal
     */
    closeBrowserModal(modal) {
        modal.remove();
    }
    
    /**
     * Show browser error
     */
    showBrowserError(message) {
        const fileList = document.querySelector('#fileList');
        fileList.innerHTML = `
            <div class="browser-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error: ${message}</p>
                <button class="btn btn-secondary" onclick="this.parentElement.remove()">
                    <i class="fas fa-times"></i>
                    Dismiss
                </button>
            </div>
        `;
    }
    
    /**
     * Start the installation process
     */
    async startInstallation() {
        const path = this.installationPath.value.trim();
        
        if (!path) {
            alert('Please enter an installation directory path');
            return;
        }
        
        if (!confirm(`LeRobot will be installed to:\n\n${path}\n\nThis will:\n• Clone the LeRobot repository\n• Create a new conda environment\n• Install all required dependencies\n\nContinue?`)) {
            return;
        }
        
        try {
            const response = await fetch('/api/start-installation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    installation_path: path
                })
            });
            
            if (response.ok) {
                this.setInstallationState(true);
                this.clearLog();
                this.addLogMessage('Starting LeRobot installation...', 'info');
            } else {
                const error = await response.json();
                this.addLogMessage(`Failed to start installation: ${error.error}`, 'error');
            }
            
        } catch (error) {
            console.error('Failed to start installation:', error);
            this.addLogMessage('Failed to communicate with installation service', 'error');
        }
    }
    
    /**
     * Cancel the installation process
     */
    async cancelInstallation() {
        if (!confirm('Are you sure you want to cancel the installation?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/cancel-installation', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.addLogMessage('Cancelling installation...', 'warning');
            }
            
        } catch (error) {
            console.error('Failed to cancel installation:', error);
            this.addLogMessage('Failed to cancel installation', 'error');
        }
    }
    
    /**
     * Set UI state for installation active/inactive
     */
    setInstallationState(active) {
        this.installationRunning = active;
        
        this.startInstallBtn.disabled = active;
        this.cancelInstallBtn.disabled = !active;
        this.installationPath.disabled = active;
        this.browseBtn.disabled = active;
        
        if (!active) {
            this.resetSteps();
            this.updateProgress(0);
        }
    }
    
    /**
     * Handle log messages from backend
     */
    handleLogMessage(message) {
        let messageType = 'info';
        
        if (message.toLowerCase().includes('error')) {
            messageType = 'error';
        } else if (message.toLowerCase().includes('warning')) {
            messageType = 'warning';
        } else if (message.toLowerCase().includes('successfully') || 
                   message.toLowerCase().includes('completed')) {
            messageType = 'success';
        } else if (message.toLowerCase().includes('executing:') || 
                   message.toLowerCase().includes('running:')) {
            messageType = 'command';
        }
        
        this.addLogMessage(message, messageType);
    }
    
    /**
     * Handle progress updates from backend
     */
    handleProgressUpdate(data) {
        this.updateProgress(data.progress);
        this.updateStatus('running', data.description);
        this.updateStep(data.step, 'active');
    }
    
    /**
     * Handle error messages from backend
     */
    handleErrorMessage(message) {
        this.addLogMessage(`ERROR: ${message}`, 'error');
        this.updateStatus('failed', 'Installation failed');
        this.setInstallationState(false);
        this.showErrorModal(message);
    }
    
    /**
     * Handle port detection start during installation
     */
    handlePortDetectionStart(data) {
        this.addLogMessage('Setting up robotic arm connections...', 'info');
        this.updateProgress(90); // Near completion
        this.updateStep('detecting_ports', 'active');
        
        // Show port detection modal
        this.showPortDetectionModal();
    }
    
    /**
     * Handle installation completion
     */
    handleInstallationComplete(message) {
        this.addLogMessage(message, 'success');
        this.updateStatus('completed', 'Installation completed successfully!');
        this.updateProgress(100);
        this.markAllStepsCompleted();
        this.setInstallationState(false);
        this.showSuccessModal();
    }
    
    /**
     * Handle status updates from backend
     */
    handleStatusUpdate(data) {
        if (data.is_running) {
            this.setInstallationState(true);
        }
    }
    
    /**
     * Add a log message to the log container
     */
    addLogMessage(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-message ${type}`;
        logEntry.textContent = message;
        
        this.logContainer.appendChild(logEntry);
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }
    
    /**
     * Clear all log messages
     */
    clearLog() {
        this.logContainer.innerHTML = '';
        this.addLogMessage('Log cleared', 'info');
    }
    
    /**
     * Update progress bar
     */
    updateProgress(percent) {
        this.progressFill.style.width = `${percent}%`;
        this.progressPercent.textContent = `${percent}%`;
    }
    
    /**
     * Update status indicator
     */
    updateStatus(status, text) {
        // Remove all status classes
        this.statusIndicator.className = 'status-indicator';
        
        // Add new status class
        this.statusIndicator.classList.add(`status-${status}`);
        this.statusText.textContent = text;
    }
    
    /**
     * Update installation step
     */
    updateStep(stepId, state) {
        const steps = this.stepsContainer.querySelectorAll('.step');
        
        // Reset all steps to default state
        steps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
        
        // Find and update the current step
        const currentStep = this.stepsContainer.querySelector(`[data-step="${stepId}"]`);
        if (currentStep) {
            currentStep.classList.add(state);
            
            // Mark previous steps as completed
            const stepElements = Array.from(steps);
            const currentIndex = stepElements.indexOf(currentStep);
            
            for (let i = 0; i < currentIndex; i++) {
                stepElements[i].classList.add('completed');
            }
        }
    }
    
    /**
     * Mark all steps as completed
     */
    markAllStepsCompleted() {
        const steps = this.stepsContainer.querySelectorAll('.step');
        steps.forEach(step => {
            step.classList.remove('active');
            step.classList.add('completed');
        });
    }
    
    /**
     * Reset all steps to default state
     */
    resetSteps() {
        const steps = this.stepsContainer.querySelectorAll('.step');
        steps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }
    
    /**
     * Show success modal
     */
    showSuccessModal() {
        this.successModal.classList.add('show');
    }
    
    /**
     * Hide success modal
     */
    hideSuccessModal() {
        this.successModal.classList.remove('show');
    }
    
    /**
     * Show error modal
     */
    showErrorModal(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.add('show');
    }
    
    /**
     * Hide error modal
     */
    hideErrorModal() {
        this.errorModal.classList.remove('show');
    }
    
    /**
     * Show port detection modal during installation
     */
    showPortDetectionModal() {
        // Create and show port detection guidance modal
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.id = 'portDetectionModal';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <i class="fas fa-usb success-icon"></i>
                    <h3>Robotic Arm Setup</h3>
                </div>
                <div class="modal-body">
                    <p><strong>Final Step:</strong> Let's identify your robotic arm USB ports!</p>
                    <div class="instructions">
                        <h4>What you'll need to do:</h4>
                        <ol>
                            <li>Make sure both leader and follower arms are connected via USB</li>
                            <li>Follow the guided instructions to identify each arm's port</li>
                            <li>The system will automatically save your configuration</li>
                        </ol>
                    </div>
                    <p class="success-note">This will only take a few minutes and makes your robots ready to use!</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" id="skipPortDetection">
                        <i class="fas fa-forward"></i>
                        Skip (Configure Later)
                    </button>
                    <button type="button" class="btn btn-primary" id="startPortDetection">
                        <i class="fas fa-usb"></i>
                        Set Up Arm Ports
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Bind events
        const skipBtn = modal.querySelector('#skipPortDetection');
        const startBtn = modal.querySelector('#startPortDetection');
        
        skipBtn.addEventListener('click', () => {
            this.skipPortDetection(modal);
        });
        
        startBtn.addEventListener('click', () => {
            this.openPortDetectionInterface(modal);
        });
    }
    
    /**
     * Skip port detection and complete installation
     */
    skipPortDetection(modal) {
        modal.remove();
        this.addLogMessage('Port detection skipped. You can configure ports later.', 'warning');
        this.updateStep('detecting_ports', 'completed');
        this.handleInstallationComplete('Installation completed! Port configuration can be done later.');
    }
    
    /**
     * Open port detection interface
     */
    openPortDetectionInterface(modal) {
        modal.remove();
        this.addLogMessage('Opening port detection interface...', 'info');
        
        // Open port detection in new window/tab
        const portDetectionWindow = window.open('port-detection.html', '_blank', 'width=1000,height=700');
        
        if (portDetectionWindow) {
            this.addLogMessage('Please complete port detection in the new window.', 'info');
            this.addLogMessage('Installation will complete automatically when ports are configured.', 'info');
            
            // Monitor for port detection completion
            this.monitorPortDetection(portDetectionWindow);
        } else {
            // Fallback: redirect current window
            this.addLogMessage('Redirecting to port detection...', 'info');
            window.location.href = 'port-detection.html';
        }
    }
    
    /**
     * Monitor port detection completion
     */
    monitorPortDetection(portWindow) {
        const checkInterval = setInterval(() => {
            // Check if port detection window is closed or completed
            if (portWindow.closed) {
                clearInterval(checkInterval);
                this.addLogMessage('Port detection window closed.', 'info');
                this.updateStep('detecting_ports', 'completed');
                this.handleInstallationComplete('Installation completed successfully!');
            }
        }, 1000);
        
        // Listen for port configuration saved event
        this.socket.on('port_config_saved', (data) => {
            clearInterval(checkInterval);
            this.addLogMessage(`Arm ports configured: Leader: ${data.leader_port}, Follower: ${data.follower_port}`, 'success');
            this.updateStep('detecting_ports', 'completed');
            
            // Close port detection window if still open
            if (!portWindow.closed) {
                portWindow.close();
            }
            
            this.handleInstallationComplete('Installation completed! Your robotic arms are ready to use.');
        });
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.installationApp = new InstallationApp();
});