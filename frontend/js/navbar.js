// Mobile menu toggle functionality
document.addEventListener('DOMContentLoaded', function() {
  const mobileMenuToggle = document.getElementById('mobileMenuToggle');
  const mobileMenu = document.getElementById('mobileMenu');
  
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
}); 