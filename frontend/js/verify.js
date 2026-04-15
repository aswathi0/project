// frontend/js/verify.js
let currentDocumentId = null;

// Handle file upload
document.getElementById('fileInput').addEventListener('change', handleFileSelect);
document.getElementById('uploadArea').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// Drag and drop
const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (file.size > 10 * 1024 * 1024) {
        showToast('File too large. Max 10MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('previewArea').style.display = 'block';
        document.getElementById('resultsArea').style.display = 'none';
        currentDocumentId = null;
    };
    reader.readAsDataURL(file);

    // Store file for upload
    window.selectedFile = file;
}

function clearPreview() {
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('previewArea').style.display = 'none';
    document.getElementById('resultsArea').style.display = 'none';
    document.getElementById('fileInput').value = '';
    window.selectedFile = null;
    currentDocumentId = null;
}

async function startVerification() {
    if (!window.selectedFile) {
        showToast('Please select a file first', 'error');
        return;
    }

    // Show loading
    document.getElementById('previewArea').style.display = 'none';
    document.getElementById('loadingIndicator').style.display = 'block';

    // Update loading steps
    const loadingSteps = ['Extracting LBP features...', 'Analyzing GLCM texture...',
        'Running GAN anomaly detection...', 'Computing final score...'];
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        if (stepIndex < loadingSteps.length) {
            document.querySelector('.loading-step').textContent = loadingSteps[stepIndex];
            stepIndex++;
        }
    }, 2000);

    try {
        // Upload document
        const uploadResult = await api.uploadDocument(window.selectedFile);

        if (!uploadResult.id) {
            throw new Error('Upload failed');
        }

        currentDocumentId = uploadResult.id;

        // Verify document
        const verifyResult = await api.verifyDocument(currentDocumentId);

        clearInterval(stepInterval);
        document.getElementById('loadingIndicator').style.display = 'none';

        // Display results
        displayResults(verifyResult);

    } catch (error) {
        clearInterval(stepInterval);
        document.getElementById('loadingIndicator').style.display = 'none';
        document.getElementById('previewArea').style.display = 'block';
        showToast('Verification failed: ' + error.message, 'error');
    }
}

function displayResults(result) {
    const resultContent = document.getElementById('resultContent');
    const isFake = result.is_fake;
    const confidence = (result.confidence * 100).toFixed(1);
    const finalScore = result.final_score.toFixed(3);
    const textureScore = result.texture_score?.toFixed(3) || 'N/A';
    const ganScore = result.gan_score?.toFixed(3) || 'N/A';
    const processingTime = result.processing_time_ms;

    resultContent.innerHTML = `
        <div class="result-card">
            <div class="result-status ${isFake ? 'fake' : 'real'}">
                <div class="result-status-icon">${isFake ? '⚠️' : '✅'}</div>
                <div class="result-status-text">
                    ${isFake ? 'FAKE Certificate Detected' : 'Authentic Certificate'}
                </div>
            </div>
            
            <div class="result-confidence" style="color: ${isFake ? 'var(--danger-color)' : 'var(--secondary-color)'}">
                ${confidence}% Confidence
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">Final Score</div>
                    <div class="detail-value">${finalScore}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Texture Score</div>
                    <div class="detail-value">${textureScore}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">GAN Score</div>
                    <div class="detail-value">${ganScore}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Processing Time</div>
                    <div class="detail-value">${processingTime}ms</div>
                </div>
            </div>
        </div>
        
        <div class="result-card">
            <h4>Texture Analysis Details</h4>
            ${result.texture_features ? `
                <div class="result-details">
                    <div class="detail-item">
                        <div class="detail-label">LBP Entropy</div>
                        <div class="detail-value">${result.texture_features.lbp_entropy?.toFixed(4) || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">GLCM Contrast</div>
                        <div class="detail-value">${result.texture_features.glm_contrast?.toFixed(4) || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Gabor Energy</div>
                        <div class="detail-value">${result.texture_features.gabor_energy?.toFixed(4) || 'N/A'}</div>
                    </div>
                </div>
            </div>
            ` : '<p>No texture features available</p>'}
        </div>
    `;

    document.getElementById('resultsArea').style.display = 'block';

    // Save to history
    saveToHistory(result);
}

function saveToHistory(result) {
    let history = JSON.parse(localStorage.getItem('verification_history') || '[]');
    history.unshift({
        id: Date.now(),
        document_id: currentDocumentId,
        is_fake: result.is_fake,
        confidence: result.confidence,
        timestamp: new Date().toISOString()
    });
    // Keep last 50 records
    history = history.slice(0, 50);
    localStorage.setItem('verification_history', JSON.stringify(history));
}

function resetVerification() {
    clearPreview();
    document.getElementById('resultsArea').style.display = 'none';
    window.selectedFile = null;
    currentDocumentId = null;
}

// Attach verify button event
document.getElementById('verifyBtn')?.addEventListener('click', startVerification);