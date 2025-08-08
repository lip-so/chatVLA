// Path helper for handling navigation in both local and production environments
const PathHelper = {
  // Get the base path for the application
  getBasePath() {
    // Check if we're in a subdirectory deployment
    const pathname = window.location.pathname;
    
    // If we're on a page in /pages/, we're one level deep
    if (pathname.includes('/pages/')) {
      return '../';
    }
    // If we're on a page in /frontend/pages/, we're also one level deep from frontend
    else if (pathname.includes('/frontend/pages/')) {
      return '../';
    }
    // If we're at the root or in /frontend/
    else {
      return './';
    }
  },
  
  // Get absolute path for navigation
  getAbsolutePath(path) {
    // Remove leading slash if present
    path = path.replace(/^\//, '');
    
    // For production, we need to handle the base URL properly
    const origin = window.location.origin;
    const basePath = this.getBasePath();
    
    // If the path already includes 'frontend', don't add it again
    if (path.startsWith('frontend/')) {
      return origin + '/' + path;
    }
    
    // Otherwise, construct the proper path
    return origin + '/frontend/' + path;
  },
  
  // Navigate to a page
  navigateTo(path) {
    window.location.href = this.getAbsolutePath(path);
  },
  
  // Get the correct path for resources (CSS, JS, images)
  getResourcePath(resource) {
    const basePath = this.getBasePath();
    return basePath + resource;
  }
};

// Export for use in other scripts
window.PathHelper = PathHelper;