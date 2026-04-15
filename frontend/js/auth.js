// frontend/js/auth.js
// Check if user is logged in
function isLoggedIn() {
    const token = localStorage.getItem('access_token');
    const user = localStorage.getItem('user');
    return token && user;
}

// Get current user
function getCurrentUser() {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
}

// Update navbar based on auth state
function updateNavbar() {
    const loginBtn = document.getElementById('loginBtn');
    const registerBtn = document.getElementById('registerBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const logoutBtnNav = document.getElementById('logoutBtnNav');

    if (isLoggedIn()) {
        if (loginBtn) loginBtn.style.display = 'none';
        if (registerBtn) registerBtn.style.display = 'none';
        if (logoutBtn) logoutBtn.style.display = 'inline-block';
        if (logoutBtnNav) logoutBtnNav.style.display = 'inline-block';

        // Update username in dashboard if present
        const usernameSpan = document.getElementById('username');
        if (usernameSpan) {
            const user = getCurrentUser();
            usernameSpan.textContent = user?.username || 'User';
        }
    } else {
        if (loginBtn) loginBtn.style.display = 'inline-block';
        if (registerBtn) registerBtn.style.display = 'inline-block';
        if (logoutBtn) logoutBtn.style.display = 'none';
        if (logoutBtnNav) logoutBtnNav.style.display = 'none';
    }
}

// Logout function
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    showToast('Logged out successfully');
    setTimeout(() => {
        window.location.href = 'index.html';
    }, 1000);
}

// Handle login form
if (document.getElementById('loginForm')) {
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const data = await api.login(email, password);

            if (data.access_token) {
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('user', JSON.stringify(data.user));
                showToast('Login successful!');
                setTimeout(() => {
                    window.location.href = 'dashboard.html';
                }, 1000);
            } else {
                showToast(data.detail || 'Login failed', 'error');
            }
        } catch (error) {
            showToast('Login failed. Please try again.', 'error');
        }
    });
}

// Handle register form
if (document.getElementById('registerForm')) {
    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = document.getElementById('username').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        if (password !== confirmPassword) {
            showToast('Passwords do not match', 'error');
            return;
        }

        try {
            const data = await api.register(username, email, password);

            if (data.access_token) {
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('user', JSON.stringify(data.user));
                showToast('Registration successful!');
                setTimeout(() => {
                    window.location.href = 'dashboard.html';
                }, 1000);
            } else {
                showToast(data.detail || 'Registration failed', 'error');
            }
        } catch (error) {
            showToast('Registration failed. Please try again.', 'error');
        }
    });
}

// Handle logout buttons
document.addEventListener('DOMContentLoaded', () => {
    updateNavbar();

    const logoutBtns = document.querySelectorAll('#logoutBtn, #logoutBtnNav');
    logoutBtns.forEach(btn => {
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                logout();
            });
        }
    });

    // Redirect to login if accessing protected pages without auth
    const protectedPages = ['dashboard.html', 'verify.html', 'history.html'];
    const currentPage = window.location.pathname.split('/').pop();

    if (protectedPages.includes(currentPage) && !isLoggedIn()) {
        window.location.href = 'login.html';
    }
});