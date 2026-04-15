// frontend/js/api.js
const API_BASE_URL = 'http://localhost:8000/api';

// Get token from localStorage
function getToken() {
    return localStorage.getItem('access_token');
}

// Set auth header
function getAuthHeaders() {
    const token = getToken();
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

// API calls
const api = {
    // Auth
    async register(username, email, password) {
        const response = await fetch(`${API_BASE_URL}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });
        return response.json();
    },

    async login(email, password) {
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        return response.json();
    },

    // Documents
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/documents/upload`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${getToken()}` },
            body: formData
        });
        return response.json();
    },

    // Verification
    async verifyDocument(documentId) {
        const response = await fetch(`${API_BASE_URL}/verify`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ document_id: documentId })
        });
        return response.json();
    },

    // History
    async getVerificationHistory(limit = 50, offset = 0) {
        const response = await fetch(`${API_BASE_URL}/verify/history?limit=${limit}&offset=${offset}`, {
            headers: getAuthHeaders()
        });
        return response.json();
    },

    // Stats
    async getSystemStats() {
        const response = await fetch(`${API_BASE_URL}/stats/system`, {
            headers: getAuthHeaders()
        });
        return response.json();
    },

    // Health check
    async healthCheck() {
        const response = await fetch(`${API_BASE_URL}/health`);
        return response.json();
    },

    // Get single verification result details
    async getVerificationResult(verificationId) {
        const response = await fetch(`${API_BASE_URL}/verify/${verificationId}`, {
            headers: getAuthHeaders()
        });
        return response.json();
    }
};

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    if (!toast) {
        const newToast = document.createElement('div');
        newToast.id = 'toast';
        newToast.className = 'toast';
        document.body.appendChild(newToast);
    }

    const toastEl = document.getElementById('toast');
    toastEl.textContent = message;
    toastEl.className = `toast ${type} show`;

    setTimeout(() => {
        toastEl.classList.remove('show');
    }, 3000);
}