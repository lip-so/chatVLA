// General access control for restricted pages
document.addEventListener('DOMContentLoaded', function() {
  // Check if user is in demo mode (coming from Vision page)
  const demoMode = sessionStorage.getItem('demoMode');
  const currentPage = window.location.pathname;
  
  // Define restricted pages
  const restrictedPages = ['/pages/databench.html', '/pages/plug-and-play-databench-style.html'];
  const isRestrictedPage = restrictedPages.some(page => currentPage.includes(page.replace('/pages/', '')));
  
  if (demoMode === 'vision-only' && isRestrictedPage) {
    // Redirect to Vision page with a message
    alert('This feature is not available in demo mode. Please contact us for full access.');
            window.location.href = window.location.origin + '/frontend/pages/vision.html';
    return;
  }
  
  // If not in demo mode, allow full access
  if (!demoMode || demoMode !== 'vision-only') {
    // User has full access - no restrictions
    sessionStorage.setItem('fullAccess', 'true');
    console.log('Full access mode enabled');
  }
}); 