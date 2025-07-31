// Mobile menu toggle functionality and authentication state
document.addEventListener('DOMContentLoaded', function() {
  const mobileMenuToggle = document.getElementById('mobileMenuToggle');
  const mobileMenu = document.getElementById('mobileMenu');
  
  // Load auth.js if not already loaded
  if (!window.authManager) {
    const authScript = document.createElement('script');
    authScript.src = '/js/auth.js';
    document.head.appendChild(authScript);
  }
  
  // Update navbar based on authentication state
  function updateNavbar() {
    if (!window.authManager) return;
    
    const navLinks = document.querySelector('.nav-links');
    const mobileMenuContent = document.querySelector('.mobile-menu-content');
    
    if (!navLinks) return;
    
    // Check if user is authenticated
    const isAuthenticated = window.authManager.isAuthenticated();
    const user = window.authManager.user;
    
    // Remove existing auth-related links
    const existingAuthLinks = navLinks.querySelectorAll('.auth-link');
    existingAuthLinks.forEach(link => link.remove());
    
    if (mobileMenuContent) {
      const mobileAuthLinks = mobileMenuContent.querySelectorAll('.auth-link');
      mobileAuthLinks.forEach(link => link.remove());
    }
    
    // Create auth links
    if (isAuthenticated) {
      // Desktop navbar
      const userDisplay = document.createElement('span');
      userDisplay.className = 'auth-link user-display';
      userDisplay.style.cssText = 'color: rgba(255, 255, 255, 0.8); font-size: 14px; margin-right: 15px;';
      userDisplay.textContent = `Hi, ${user.username}`;
      
      const logoutLink = document.createElement('a');
      logoutLink.href = '#';
      logoutLink.className = 'auth-link';
      logoutLink.textContent = 'Logout';
      logoutLink.addEventListener('click', handleLogout);
      
      navLinks.appendChild(userDisplay);
      navLinks.appendChild(logoutLink);
      
      // Mobile menu
      if (mobileMenuContent) {
        const mobileUserDisplay = userDisplay.cloneNode(true);
        mobileUserDisplay.style.cssText = 'color: rgba(255, 255, 255, 0.8); font-size: 14px; display: block; padding: 10px 0;';
        
        const mobileLogoutLink = logoutLink.cloneNode(true);
        mobileLogoutLink.addEventListener('click', handleLogout);
        
        mobileMenuContent.appendChild(mobileUserDisplay);
        mobileMenuContent.appendChild(mobileLogoutLink);
      }
    } else {
      // Desktop navbar
      const loginLink = document.createElement('a');
      loginLink.href = '/pages/login.html';
      loginLink.className = 'auth-link';
      loginLink.textContent = 'Login';
      
      const registerLink = document.createElement('a');
      registerLink.href = '/pages/register.html';
      registerLink.className = 'auth-link primary-auth-link';
      registerLink.textContent = 'Sign Up';
      registerLink.style.cssText = 'background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(51, 51, 51, 0.85)); color: white; padding: 8px 16px; border-radius: 50px; margin-left: 10px; transition: all 0.3s ease;';
      
      // Add hover effect
      registerLink.addEventListener('mouseenter', () => {
        registerLink.style.transform = 'translateY(-2px) scale(1.02)';
        registerLink.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.3)';
      });
      
      registerLink.addEventListener('mouseleave', () => {
        registerLink.style.transform = 'none';
        registerLink.style.boxShadow = 'none';
      });
      
      navLinks.appendChild(loginLink);
      navLinks.appendChild(registerLink);
      
      // Mobile menu
      if (mobileMenuContent) {
        const mobileLoginLink = loginLink.cloneNode(true);
        const mobileRegisterLink = registerLink.cloneNode(true);
        mobileRegisterLink.style.cssText = '';
        
        mobileMenuContent.appendChild(mobileLoginLink);
        mobileMenuContent.appendChild(mobileRegisterLink);
      }
    }
  }
  
  // Handle logout
  async function handleLogout(e) {
    e.preventDefault();
    
    if (!window.authManager) return;
    
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          ...window.authManager.getAuthHeaders(),
          'Content-Type': 'application/json'
        }
      });
      
      window.authManager.clearAuthData();
      window.location.href = '/';
    } catch (error) {
      console.error('Logout error:', error);
      window.authManager.clearAuthData();
      window.location.href = '/';
    }
  }
  
  // Wait for auth manager to load then update navbar
  const checkAuthManager = setInterval(() => {
    if (window.authManager) {
      clearInterval(checkAuthManager);
      updateNavbar();
      
      // Verify token validity
      window.authManager.verifyToken().then(isValid => {
        if (!isValid && window.authManager.token) {
          // Token was invalid, update navbar
          updateNavbar();
        }
      });
    }
  }, 100);
  
  // Mobile menu functionality
  if (mobileMenuToggle && mobileMenu) {
    // Toggle menu on hamburger click
    mobileMenuToggle.addEventListener('click', function() {
      this.classList.toggle('active');
      mobileMenu.classList.toggle('active');
      
      // Prevent body scroll when menu is open
      if (mobileMenu.classList.contains('active')) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    });
    
    // Close mobile menu when clicking a link
    const mobileMenuLinks = mobileMenu.querySelectorAll('a');
    mobileMenuLinks.forEach(link => {
      link.addEventListener('click', () => {
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
      });
    });
    
    // Close mobile menu when clicking outside
    mobileMenu.addEventListener('click', (e) => {
      if (e.target === mobileMenu) {
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
      }
    });
    
    // Close menu on window resize if it becomes desktop size
    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) {
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
      }
    });
  }
  
  // Check authentication on protected pages
  function checkPageProtection() {
    const protectedPages = ['/pages/databench.html', '/pages/plug-and-play.html', '/pages/port-detection.html'];
    const currentPath = window.location.pathname;
    
    // Check if current page is protected
    if (protectedPages.some(page => currentPath.includes(page))) {
      // Wait for auth manager
      const checkAuth = setInterval(() => {
        if (window.authManager) {
          clearInterval(checkAuth);
          
          // Verify authentication
          if (!window.authManager.isAuthenticated()) {
            window.authManager.redirectToLogin();
          } else {
            // Verify token is still valid
            window.authManager.verifyToken().then(isValid => {
              if (!isValid) {
                window.authManager.redirectToLogin();
              }
            });
          }
        }
      }, 50);
    }
  }
  
  // Run page protection check
  checkPageProtection();
}); 