// Vision page access control
document.addEventListener('DOMContentLoaded', function() {
  // Hide all navigation links to DataBench and Plug & Play
  const restrictedNavFeatures = document.querySelectorAll('.nav-links-features, .mobile-menu-features');
  restrictedNavFeatures.forEach(element => {
    element.style.display = 'none';
  });
  
  // Remove any existing buttons or links that might lead to restricted sections
  const restrictedLinks = document.querySelectorAll('a[href*="databench"], a[href*="plug-and-play"]');
  restrictedLinks.forEach(link => {
    // Check if it's in the CTA section or other content areas (not navigation)
    const parent = link.closest('.cta-buttons, .feature-card, .demo-section');
    if (parent) {
      link.remove();
    }
  });
  
  // Update page title and meta to reflect Vision-only access
  document.title = 'Tune Robotics - Vision (Demo Access)';
  
  // Optional: Add a subtle indicator that this is demo/limited access
  const heroSection = document.querySelector('.vision-hero');
  if (heroSection) {
    const subtitle = heroSection.querySelector('.subtitle');
    if (subtitle) {
      // Add a small indicator that this is demo access
      const demoIndicator = document.createElement('div');
      demoIndicator.style.cssText = `
        font-size: 0.9rem;
        color: rgba(0, 0, 0, 0.6);
        margin-top: 1rem;
        font-style: normal;
        font-weight: 500;
      `;
      demoIndicator.textContent = '• Demo Access • Vision Features Only •';
      subtitle.parentNode.insertBefore(demoIndicator, subtitle.nextSibling);
    }
  }
  
  // Prevent access to restricted pages if someone tries to navigate directly
  window.addEventListener('beforeunload', function() {
    // Clear any session data that might grant broader access
    sessionStorage.removeItem('fullAccess');
  });
  
  // Initialize demo mode
  sessionStorage.setItem('demoMode', 'vision-only');
  
  console.log('Vision page initialized in demo mode - limited navigation enabled');
}); 