// Mobile menu toggle functionality and Firebase authentication state
document.addEventListener('DOMContentLoaded', function() {
  const mobileMenuToggle = document.getElementById('mobileMenuToggle');
  const mobileMenu = document.getElementById('mobileMenu');
  
  // Update navbar based on Firebase authentication state
  function updateNavbar() {
    const navLinks = document.querySelector('.nav-links');
    const mobileMenuContent = document.querySelector('.mobile-menu-content');
    
    if (!navLinks) return;
    
    // Check if user is authenticated
    let isAuthenticated = false;
    let currentUser = null;
    
    // Try to get Firebase auth state first
    if (window.firebaseAuth && window.firebaseAuth.currentUser) {
      isAuthenticated = true;
      currentUser = window.firebaseAuth.currentUser;
    }
    
    // Fallback: Check localStorage for development when Firebase isn't working
    if (!isAuthenticated) {
      const storedAuth = localStorage.getItem('tune_auth_state');
      const storedUser = localStorage.getItem('tune_auth_user');
      
      if (storedAuth === 'true' && storedUser) {
        try {
          isAuthenticated = true;
          currentUser = JSON.parse(storedUser);
        } catch (e) {
          // Clear invalid stored data
          localStorage.removeItem('tune_auth_state');
          localStorage.removeItem('tune_auth_user');
        }
      }
    }
    
    // Remove existing auth-related links
    const existingAuthLinks = navLinks.querySelectorAll('.auth-link');
    existingAuthLinks.forEach(link => link.remove());
    
    if (mobileMenuContent) {
      const mobileAuthLinks = mobileMenuContent.querySelectorAll('.auth-link');
      mobileAuthLinks.forEach(link => link.remove());
    }
    
    // Create auth links
    if (isAuthenticated && currentUser) {
      // Desktop navbar - only show logout button
      const logoutLink = document.createElement('a');
      logoutLink.href = '#';
      logoutLink.className = 'auth-link';
      logoutLink.textContent = 'Logout';
      logoutLink.style.cssText = 'background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(51, 51, 51, 0.85)); color: white; padding: 8px 16px; border-radius: 50px; margin-left: 10px; transition: all 0.3s ease;';
      logoutLink.addEventListener('click', handleLogout);
      
      // Add hover effect
      logoutLink.addEventListener('mouseenter', () => {
        logoutLink.style.transform = 'translateY(-2px) scale(1.02)';
        logoutLink.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.3)';
      });
      
      logoutLink.addEventListener('mouseleave', () => {
        logoutLink.style.transform = 'none';
        logoutLink.style.boxShadow = 'none';
      });
      
      navLinks.appendChild(logoutLink);
      
      // Mobile menu
      if (mobileMenuContent) {
        const mobileLogoutLink = document.createElement('a');
        mobileLogoutLink.href = '#';
        mobileLogoutLink.className = 'auth-link';
        mobileLogoutLink.textContent = 'Logout';
        mobileLogoutLink.addEventListener('click', handleLogout);
        
        mobileMenuContent.appendChild(mobileLogoutLink);
      }
    } else {
      // Desktop navbar - show login/signup buttons
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
    
    // Show/hide protected features
    const navLinksFeatures = document.querySelector('.nav-links-features');
    const mobileMenuFeatures = document.querySelector('.mobile-menu-features');
    
    if (isAuthenticated) {
      if (navLinksFeatures) navLinksFeatures.style.display = 'flex';
      if (mobileMenuFeatures) mobileMenuFeatures.style.display = 'block';
    } else {
      if (navLinksFeatures) navLinksFeatures.style.display = 'none';
      if (mobileMenuFeatures) mobileMenuFeatures.style.display = 'none';
    }
  }
  
  // Handle logout with Firebase
  async function handleLogout(e) {
    e.preventDefault();
    
    if (window.firebaseAuth) {
      try {
        // Import signOut dynamically
        const { signOut } = await import('https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js');
        await signOut(window.firebaseAuth);
      } catch (error) {
        console.error('Logout error:', error);
      }
    }
    
    // Clear fallback localStorage auth data
    localStorage.removeItem('tune_auth_state');
    localStorage.removeItem('tune_auth_user');
    
    // Update UI before redirect
    if (window.updateLandingActions) {
      window.updateLandingActions();
    }
    
    // Always redirect to home
    window.location.href = '/';
  }
  
  // Initialize navbar immediately
  updateNavbar();
  
  // Wait for Firebase auth to load then update navbar
  const checkFirebaseAuth = setInterval(() => {
    if (window.firebaseAuth) {
      clearInterval(checkFirebaseAuth);
      
      // Listen for auth state changes
      if (window.onAuthStateChanged) {
        window.onAuthStateChanged(window.firebaseAuth, (user) => {
          updateNavbar();
        });
      }
      
      // Update again when Firebase is ready
      updateNavbar();
    }
  }, 100);
  
  // Fallback: show buttons after 2 seconds if Firebase never loads
  setTimeout(() => {
    if (!window.firebaseAuth) {
      console.log('Firebase not loaded, using fallback navbar');
      updateNavbar();
    }
  }, 2000);
  
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
  
  // Check authentication on protected pages (disabled for development)
  function checkPageProtection() {
    const protectedPages = ['/pages/databench.html', '/pages/plug-and-play-databench-style.html', '/pages/port-detection.html'];
    const currentPath = window.location.pathname;
    
    // Check if current page is protected
    if (protectedPages.some(page => currentPath.includes(page))) {
      console.log('On protected page:', currentPath, '- Auth protection disabled for development');
      
      // TODO: Re-enable when Firebase credentials are properly configured
      // For now, allow access to protected pages without authentication for development
      /*
      // Wait for Firebase auth
      const checkAuth = setInterval(() => {
        if (window.firebaseAuth) {
          clearInterval(checkAuth);
          
          // Check authentication state
          if (window.onAuthStateChanged) {
            window.onAuthStateChanged(window.firebaseAuth, (user) => {
              if (!user) {
                // User not authenticated, redirect to login
                window.location.href = '/frontend/pages/login.html?redirect=' + encodeURIComponent(window.location.pathname);
              }
            });
          }
          
          // Check immediate state
          if (!window.firebaseAuth.currentUser) {
            window.location.href = '/frontend/pages/login.html?redirect=' + encodeURIComponent(window.location.pathname);
          }
        }
      }, 50);
      
      // Fallback: if Firebase doesn't load in 3 seconds, redirect to login
      setTimeout(() => {
        if (!window.firebaseAuth || !window.firebaseAuth.currentUser) {
          window.location.href = '/frontend/pages/login.html?redirect=' + encodeURIComponent(window.location.pathname);
        }
      }, 3000);
      */
    }
  }
  
  // Run page protection check
  checkPageProtection();
  
  // Make updateNavbar available globally so other pages can trigger updates
  window.updateNavbar = updateNavbar;
}); 