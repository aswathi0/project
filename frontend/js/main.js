// frontend/js/main.js
// Load stats on homepage
async function loadHomepageStats() {
    if (!isLoggedIn()) return;

    try {
        const stats = await api.getSystemStats();

        const totalEl = document.getElementById('totalVerifications');
        const accuracyEl = document.getElementById('accuracyRate');
        const avgTimeEl = document.getElementById('avgTime');

        if (totalEl) totalEl.textContent = stats.total_verifications || 0;
        if (accuracyEl) accuracyEl.textContent = `${((stats.fake_detection_rate || 0) * 100).toFixed(1)}%`;
        if (avgTimeEl) avgTimeEl.textContent = `${stats.average_processing_time_ms?.toFixed(0) || 0}ms`;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function toggleMobileMenu() {
    const navLinks = document.getElementById('navLinks');
    navLinks.classList.toggle('active');
}

// Load stats on homepage
document.addEventListener('DOMContentLoaded', () => {
    loadHomepageStats();

    // Check if we're on homepage
    if (window.location.pathname.includes('index.html') || window.location.pathname === '/' || window.location.pathname === '') {
        loadHomepageStats();
    }
});