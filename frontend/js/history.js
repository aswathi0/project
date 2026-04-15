// frontend/js/history.js
let currentPage = 1;
let totalPages = 1;
let allHistory = [];

async function loadHistory(page = 1) {
    const limit = 10;
    const offset = (page - 1) * limit;

    try {
        const history = await api.getVerificationHistory(limit, offset);
        allHistory = history;

        // Apply filters
        applyFilters();

    } catch (error) {
        console.error('Error loading history:', error);
        showToast('Failed to load history', 'error');
    }
}

function applyFilters() {
    const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
    const filterResult = document.getElementById('filterResult')?.value || 'all';

    let filtered = [...allHistory];

    // Apply search
    if (searchTerm) {
        filtered = filtered.filter(item =>
            item.filename.toLowerCase().includes(searchTerm)
        );
    }

    // Apply result filter
    if (filterResult === 'real') {
        filtered = filtered.filter(item => !item.is_fake);
    } else if (filterResult === 'fake') {
        filtered = filtered.filter(item => item.is_fake);
    }

    // Update pagination
    totalPages = Math.ceil(filtered.length / 10);
    const start = (currentPage - 1) * 10;
    const paginated = filtered.slice(start, start + 10);

    displayHistory(paginated);
    updatePagination();
}

function displayHistory(history) {
    const tbody = document.getElementById('historyTableBody');

    if (!history || history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No verifications found</td></tr>';
        return;
    }

    tbody.innerHTML = history.map(item => `
        <tr>
            <td>${item.id}</td>
            <td>${escapeHtml(item.filename)}</td>
            <td>
                <span class="badge-${item.is_fake ? 'fake' : 'real'}">
                    ${item.is_fake ? 'FAKE' : 'Authentic'}
                </span>
            </td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
            <td>${item.texture_score?.toFixed(3) || 'N/A'}</td>
            <td>${item.gan_score?.toFixed(3) || 'N/A'}</td>
            <td>${new Date(item.verification_time).toLocaleString()}</td>
            <td>
                <button onclick="viewVerificationDetail(${item.id})" class="btn-view">Details</button>
            </td>
        </tr>
    `).join('');
}

function updatePagination() {
    const paginationDiv = document.getElementById('pagination');
    if (!paginationDiv) return;

    let html = '';
    for (let i = 1; i <= totalPages; i++) {
        html += `<button class="${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }
    paginationDiv.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    applyFilters();
}

async function viewVerificationDetail(verificationId) {
    // Show modal with details
    try {
        const result = await api.getVerificationResult(verificationId);
        showDetailModal(result);
    } catch (error) {
        showToast('Failed to load details', 'error');
    }
}

function showDetailModal(result) {
    const modalHtml = `
        <div id="detailModal" class="modal">
            <div class="modal-content">
                <span class="modal-close" onclick="closeModal()">&times;</span>
                <h2>Verification Details</h2>
                <div class="modal-body">
                    <p><strong>Document:</strong> ${result.document?.filename}</p>
                    <p><strong>Result:</strong> ${result.verification?.is_fake ? 'FAKE' : 'Authentic'}</p>
                    <p><strong>Confidence:</strong> ${(result.verification?.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Final Score:</strong> ${result.verification?.final_score?.toFixed(3)}</p>
                    <p><strong>Texture Score:</strong> ${result.verification?.texture_score?.toFixed(3)}</p>
                    <p><strong>GAN Score:</strong> ${result.verification?.gan_score?.toFixed(3)}</p>
                    <p><strong>Processing Time:</strong> ${result.verification?.processing_time_ms}ms</p>
                    <p><strong>Date:</strong> ${new Date(result.verification?.verification_time).toLocaleString()}</p>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existingModal = document.getElementById('detailModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    document.getElementById('detailModal').style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('detailModal');
    if (modal) modal.remove();
}

// Add CSS for modal
const modalStyle = document.createElement('style');
modalStyle.textContent = `
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    .modal-content {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        max-width: 500px;
        width: 90%;
        position: relative;
    }
    .modal-close {
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 1.5rem;
        cursor: pointer;
    }
    .btn-view {
        padding: 0.25rem 0.75rem;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
`;
document.head.appendChild(modalStyle);

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();

    const searchInput = document.getElementById('searchInput');
    const filterSelect = document.getElementById('filterResult');

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            currentPage = 1;
            applyFilters();
        });
    }

    if (filterSelect) {
        filterSelect.addEventListener('change', () => {
            currentPage = 1;
            applyFilters();
        });
    }
});