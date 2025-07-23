/**
 * JavaScript for USB Port Detection Interface
 * Provides interactive guidance for identifying robotic arm USB ports
 */

class PortDetectionApp {
    constructor() {
        this.socket = null;
        this.currentStep = 'setup';
        this.detectedPorts = [];
        this.leaderPort = null;
        this.followerPort = null;
        this.initialPorts = [];
        
        this.initializeElements();
        this.bindEvents();
        this.connectSocket();
        this.loadAvailablePorts();
    }
    
    /**
     * Initialize DOM element references
     */
    initializeElements() {
        // Main interface elements
        this.detectionStatus = document.getElementById('detectionStatus');
        this.statusText = document.getElementById('statusText');
        this.currentStep = document.getElementById('currentStep');
        this.stepIcon = document.getElementById('stepIcon');
        this.stepTitle = document.getElementById('stepTitle');
        this.stepDescription = document.getElementById('stepDescription');
        this.stepActions = document.getElementById('stepActions');
        this.progressSteps = document.getElementById('progressSteps');
        
        // Buttons
        this.startDetectionBtn = document.getElementById('startDetectionBtn');
        this.refreshPortsBtn = document.getElementById('refreshPortsBtn');
        this.saveConfigBtn = document.getElementById('saveConfigBtn');
        this.restartDetectionBtn = document.getElementById('restartDetectionBtn');
        
        // Ports display
        this.portsList = document.getElementById('portsList');
        
        // Results
        this.resultsCard = document.getElementById('resultsCard');
        this.leaderPortEl = document.getElementById('leaderPort');
        this.followerPortEl = document.getElementById('followerPort');
        this.leaderStatus = document.getElementById('leaderStatus');
        this.followerStatus = document.getElementById('followerStatus');
        this.resultsActions = document.getElementById('resultsActions');
        
        // Modals
        this.instructionModal = document.getElementById('instructionModal');
        this.modalStepIndicator = document.getElementById('modalStepIndicator');
        this.modalTitle = document.getElementById('modalTitle');
        this.modalInstruction = document.getElementById('modalInstruction');
        this.modalDetails = document.getElementById('modalDetails');
        this.instructionVisual = document.getElementById('instructionVisual');
        this.instructionCountdown = document.getElementById('instructionCountdown');
        this.countdownText = document.getElementById('countdownText');
        this.nextStepBtn = document.getElementById('nextStepBtn');
        this.skipStepBtn = document.getElementById('skipStepBtn');
        
        this.successModal = document.getElementById('successModal');
        this.successLeaderPort = document.getElementById('successLeaderPort');
        this.successFollowerPort = document.getElementById('successFollowerPort');
        this.closeSuccessBtn = document.getElementById('closeSuccessBtn');
    }
    
    /**
     * Bind event handlers
     */
    bindEvents() {
        this.startDetectionBtn.addEventListener('click', () => this.startDetection());
        this.refreshPortsBtn.addEventListener('click', () => this.loadAvailablePorts());
        this.saveConfigBtn.addEventListener('click', () => this.saveConfiguration());
        this.restartDetectionBtn.addEventListener('click', () => this.restartDetection());
        
        this.nextStepBtn.addEventListener('click', () => this.handleNextStep());
        this.skipStepBtn.addEventListener('click', () => this.skipCurrentStep());
        this.closeSuccessBtn.addEventListener('click', () => this.hideSuccessModal());
        
        // Close modals on outside click
        this.instructionModal.addEventListener('click', (e) => {
            if (e.target === this.instructionModal) {
                // Don't allow closing during detection
            }
        });
        
        this.successModal.addEventListener('click', (e) => {
            if (e.target === this.successModal) this.hideSuccessModal();
        });
        
        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideSuccessModal();
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
                console.log('Connected to port detection service');
            });
            
            this.socket.on('port_list_update', (data) => {
                this.handlePortListUpdate(data.ports);
            });
            
            this.socket.on('port_change_detected', (data) => {
                this.handlePortChangeDetected(data);
            });
            
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
        }
    }
    
    /**
     * Load available ports from backend
     */
    async loadAvailablePorts() {
        try {
            this.portsList.innerHTML = '<div class="loading-message"><i class="fas fa-spinner fa-spin"></i>Loading available ports...</div>';
            
            const response = await fetch('/api/list-ports');
            const data = await response.json();
            
            this.detectedPorts = data.ports || [];
            this.displayPorts(this.detectedPorts);
            
        } catch (error) {
            console.error('Failed to load ports:', error);
            this.portsList.innerHTML = '<div class="loading-message"><i class="fas fa-exclamation-triangle"></i>Failed to load ports</div>';
        }
    }
    
    /**
     * Display ports in the UI
     */
    displayPorts(ports) {
        if (!ports || ports.length === 0) {
            this.portsList.innerHTML = '<div class="loading-message"><i class="fas fa-info-circle"></i>No serial ports detected</div>';
            return;
        }
        
        const portsHtml = ports.map(port => {
            let statusClass = 'available';
            let statusText = 'Available';
            
            if (port.device === this.leaderPort) {
                statusClass = 'leader';
                statusText = 'Leader Arm';
            } else if (port.device === this.followerPort) {
                statusClass = 'follower';
                statusText = 'Follower Arm';
            }
            
            return `
                <div class="port-item ${statusClass === 'available' ? '' : 'detected'}">
                    <div class="port-info">
                        <h4>${port.device}</h4>
                        <div class="port-details">
                            ${port.description || 'Unknown device'}
                            ${port.vid && port.pid ? ` • VID:PID ${port.vid}:${port.pid}` : ''}
                        </div>
                    </div>
                    <span class="port-status ${statusClass}">${statusText}</span>
                </div>
            `;
        }).join('');
        
        this.portsList.innerHTML = portsHtml;
    }
    
    /**
     * Start the port detection process
     */
    async startDetection() {
        try {
            // Store initial ports
            await this.loadAvailablePorts();
            this.initialPorts = [...this.detectedPorts];
            
            // Update UI state
            this.updateDetectionStatus('Starting detection process...');
            this.setActiveStep('setup');
            
            // Show initial instruction modal
            this.showInstructionModal(
                'Step 1: Initial Setup',
                'Verify Both Arms Connected',
                'Please ensure both your leader and follower robotic arms are connected to your computer via USB cables before proceeding.',
                'We will now begin the identification process by asking you to disconnect and reconnect each arm.',
                '<i class="fas fa-usb"></i>'
            );
            
        } catch (error) {
            console.error('Failed to start detection:', error);
            this.updateDetectionStatus('Failed to start detection');
        }
    }
    
    /**
     * Handle next step in the detection process
     */
    async handleNextStep() {
        switch (this.currentStep) {
            case 'setup':
                await this.startLeaderDetection();
                break;
            case 'leader_unplug':
                await this.detectLeaderDisconnection();
                break;
            case 'leader_replug':
                await this.detectLeaderReconnection();
                break;
            case 'follower_unplug':
                await this.detectFollowerDisconnection();
                break;
            case 'follower_replug':
                await this.detectFollowerReconnection();
                break;
            default:
                this.hideInstructionModal();
                break;
        }
    }
    
    /**
     * Start leader arm detection
     */
    async startLeaderDetection() {
        this.currentStep = 'leader_unplug';
        this.setActiveStep('leader');
        
        this.showInstructionModal(
            'Step 2: Leader Arm Detection',
            'Disconnect Leader Arm',
            'Please UNPLUG the USB cable from your LEADER ARM only.',
            'Make sure to only disconnect the leader arm. Leave the follower arm connected. Wait for the system to detect the disconnection.',
            '<i class="fas fa-plug" style="color: #ef4444;"></i>'
        );
        
        // Start monitoring for port changes
        this.startPortMonitoring();
    }
    
    /**
     * Detect leader arm disconnection
     */
    async detectLeaderDisconnection() {
        this.updateDetectionStatus('Waiting for leader arm disconnection...');
        this.startCountdown();
        
        // In a real implementation, this would be handled by WebSocket events
        // For now, we'll simulate the detection after a delay
        setTimeout(() => {
            this.currentStep = 'leader_replug';
            this.showInstructionModal(
                'Step 2: Leader Arm Detection',
                'Reconnect Leader Arm',
                'Great! Leader arm disconnection detected.',
                'Now please RECONNECT the USB cable to your LEADER ARM. The system will identify this as the leader port.',
                '<i class="fas fa-plug" style="color: #10b981;"></i>'
            );
        }, 3000);
    }
    
    /**
     * Detect leader arm reconnection
     */
    async detectLeaderReconnection() {
        this.updateDetectionStatus('Waiting for leader arm reconnection...');
        this.startCountdown();
        
        // Simulate detection
        setTimeout(() => {
            this.leaderPort = '/dev/cu.usbmodem14201'; // Example port
            this.updateResults('leader', this.leaderPort);
            this.startFollowerDetection();
        }, 3000);
    }
    
    /**
     * Start follower arm detection
     */
    async startFollowerDetection() {
        this.currentStep = 'follower_unplug';
        this.setActiveStep('follower');
        
        this.showInstructionModal(
            'Step 3: Follower Arm Detection',
            'Disconnect Follower Arm',
            'Please UNPLUG the USB cable from your FOLLOWER ARM only.',
            'Make sure to only disconnect the follower arm. The leader arm should remain connected. Wait for the system to detect the disconnection.',
            '<i class="fas fa-plug" style="color: #ef4444;"></i>'
        );
    }
    
    /**
     * Detect follower arm disconnection
     */
    async detectFollowerDisconnection() {
        this.updateDetectionStatus('Waiting for follower arm disconnection...');
        this.startCountdown();
        
        setTimeout(() => {
            this.currentStep = 'follower_replug';
            this.showInstructionModal(
                'Step 3: Follower Arm Detection',
                'Reconnect Follower Arm',
                'Excellent! Follower arm disconnection detected.',
                'Now please RECONNECT the USB cable to your FOLLOWER ARM. The system will identify this as the follower port.',
                '<i class="fas fa-plug" style="color: #10b981;"></i>'
            );
        }, 3000);
    }
    
    /**
     * Detect follower arm reconnection
     */
    async detectFollowerReconnection() {
        this.updateDetectionStatus('Waiting for follower arm reconnection...');
        this.startCountdown();
        
        setTimeout(() => {
            this.followerPort = '/dev/cu.usbmodem14301'; // Example port
            this.updateResults('follower', this.followerPort);
            this.completeDetection();
        }, 3000);
    }
    
    /**
     * Complete the detection process
     */
    completeDetection() {
        this.setActiveStep('complete');
        this.updateDetectionStatus('Detection completed successfully!');
        this.hideInstructionModal();
        
        // Show results
        this.resultsCard.style.display = 'block';
        this.resultsActions.style.display = 'flex';
        
        // Update ports display
        this.displayPorts(this.detectedPorts);
        
        // Show success modal
        setTimeout(() => {
            this.showSuccessModal();
        }, 500);
    }
    
    /**
     * Update detection results
     */
    updateResults(armType, port) {
        if (armType === 'leader') {
            this.leaderPortEl.textContent = port;
            this.leaderStatus.textContent = 'Detected';
            this.leaderStatus.style.color = 'var(--success-color)';
        } else if (armType === 'follower') {
            this.followerPortEl.textContent = port;
            this.followerStatus.textContent = 'Detected';
            this.followerStatus.style.color = 'var(--success-color)';
        }
    }
    
    /**
     * Set active step in progress indicator
     */
    setActiveStep(step) {
        const steps = this.progressSteps.querySelectorAll('.step-item');
        steps.forEach(stepEl => {
            stepEl.classList.remove('active', 'completed');
        });
        
        const stepMap = {
            'setup': 0,
            'leader': 1,
            'follower': 2,
            'complete': 3
        };
        
        const currentIndex = stepMap[step];
        
        // Mark previous steps as completed
        for (let i = 0; i < currentIndex; i++) {
            steps[i].classList.add('completed');
        }
        
        // Mark current step as active
        if (steps[currentIndex]) {
            steps[currentIndex].classList.add('active');
        }
    }
    
    /**
     * Update detection status
     */
    updateDetectionStatus(status) {
        this.statusText.textContent = status;
    }
    
    /**
     * Start port monitoring
     */
    startPortMonitoring() {
        // In a real implementation, this would use WebSocket events
        // to monitor port changes in real-time
        console.log('Started port monitoring');
    }
    
    /**
     * Start countdown display
     */
    startCountdown() {
        this.instructionCountdown.style.display = 'flex';
        this.nextStepBtn.style.display = 'none';
        this.skipStepBtn.style.display = 'inline-flex';
        
        let count = 5;
        const countdown = setInterval(() => {
            this.countdownText.textContent = count;
            count--;
            
            if (count < 0) {
                clearInterval(countdown);
                this.instructionCountdown.style.display = 'none';
                this.nextStepBtn.style.display = 'inline-flex';
                this.skipStepBtn.style.display = 'none';
            }
        }, 1000);
    }
    
    /**
     * Show instruction modal
     */
    showInstructionModal(stepText, title, instruction, details, visualIcon) {
        this.modalStepIndicator.textContent = stepText;
        this.modalTitle.textContent = title;
        this.modalInstruction.textContent = instruction;
        this.modalDetails.textContent = details;
        this.instructionVisual.innerHTML = `${visualIcon}`;
        
        this.instructionCountdown.style.display = 'none';
        this.nextStepBtn.style.display = 'inline-flex';
        this.skipStepBtn.style.display = 'none';
        
        this.instructionModal.classList.add('show');
    }
    
    /**
     * Hide instruction modal
     */
    hideInstructionModal() {
        this.instructionModal.classList.remove('show');
    }
    
    /**
     * Skip current step
     */
    skipCurrentStep() {
        // For demo purposes, skip to next step
        this.handleNextStep();
    }
    
    /**
     * Show success modal
     */
    showSuccessModal() {
        this.successLeaderPort.textContent = this.leaderPort || 'Not detected';
        this.successFollowerPort.textContent = this.followerPort || 'Not detected';
        this.successModal.classList.add('show');
    }
    
    /**
     * Hide success modal
     */
    hideSuccessModal() {
        this.successModal.classList.remove('show');
    }
    
    /**
     * Save configuration
     */
    async saveConfiguration() {
        try {
            const response = await fetch('/api/save-port-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    leader_port: this.leaderPort,
                    follower_port: this.followerPort
                })
            });
            
            if (response.ok) {
                alert('Configuration saved successfully!');
            } else {
                alert('Failed to save configuration');
            }
            
        } catch (error) {
            console.error('Failed to save configuration:', error);
            alert('Failed to save configuration');
        }
    }
    
    /**
     * Restart detection process
     */
    restartDetection() {
        this.leaderPort = null;
        this.followerPort = null;
        this.currentStep = 'setup';
        
        this.resultsCard.style.display = 'none';
        this.updateResults('leader', 'Not detected');
        this.updateResults('follower', 'Not detected');
        this.setActiveStep('setup');
        this.updateDetectionStatus('Ready to start');
        
        this.loadAvailablePorts();
    }
    
    /**
     * Handle port list updates from backend
     */
    handlePortListUpdate(ports) {
        this.detectedPorts = ports;
        this.displayPorts(ports);
    }
    
    /**
     * Handle port change detection from backend
     */
    handlePortChangeDetected(data) {
        console.log('Port change detected:', data);
        // Handle real-time port changes here
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.portDetectionApp = new PortDetectionApp();
});