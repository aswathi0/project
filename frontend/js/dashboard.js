// frontend/js/dashboard.js
let distributionChart = null;

async function loadDashboard() {
    try {
        // Get stats
        const stats = await api.getSystemStats();

        // Update stats cards
        document.getElementById('totalVerifications').textContent = stats.total_verifications || 0;
        document.getElementById('fakeRate').textContent = `${((stats.fake_detection_rate || 0) * 100).toFixed(1)}%`;
        document.getElementById('avgTime').textContent = `${stats.average_processing_time_ms?.toFixed(0) || 0}ms`;
        document.getElementById('avgConfidence').textContent = `${((stats.average_confidence || 0) * 100).toFixed(1)}%`;

        // Create distribution chart
        createDistributionChart(stats);

        // Load recent verifications
        await loadRecentVerifications();

    } catch (error) {
        console.error('Error loading dashboard:', error);
        showToast('Failed to load dashboard data', 'error');
    }
}

function createDistributionChart(stats) {
    const ctx = document.getElementById('distributionChart').getContext('2d');

    const total = stats.total_verifications || 1;
    const fakeCount = Math.floor(total * (stats.fake_detection_rate || 0));
    const realCount = total - fakeCount;

    if (distributionChart) {
        distributionChart.destroy();
    }

    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Authentic', 'Forged'],
            datasets: [{
                data: [realCount, fakeCount],
                backgroundColor: ['#10B981', '#EF4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

async function loadRecentVerifications() {
    try {
        const history = await api.getVerificationHistory(5);
        const tbody = document.getElementById('recentTableBody');

        if (!history || history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No verifications yet</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(item => `
            <tr>
                <td>${escapeHtml(item.filename)}</td>
                <td>
                    <span class="badge-${item.is_fake ? 'fake' : 'real'}">
                        ${item.is_fake ? 'FAKE' : 'Authentic'}
                    </span>
                </td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${new Date(item.verification_time).toLocaleDateString()}</td>
                <td>
                    <button onclick="viewDetails(${item.id})" class="btn-view">View</button>
                </td>
            </tr>
        `).join('');

    } catch (error) {
        console.error('Error loading recent verifications:', error);
    }
}

function viewDetails(verificationId) {
    // Store in session storage and redirect
    sessionStorage.setItem('view_verification_id', verificationId);
    window.location.href = 'history.html';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load dashboard on page load
document.addEventListener('DOMContentLoaded', loadDashboard);