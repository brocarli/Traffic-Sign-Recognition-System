/**
 * TrafficAI - Main JavaScript
 * Handles UI interactions and animations
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavMenu();
    initScrollAnimations();
    initDropzoneHighlight();
});

/**
 * Mobile nav menu toggle
 */
function initNavMenu() {
    const menuBtn = document.getElementById('menuBtn');
    const navLinks = document.querySelector('.nav-links');

    if (!menuBtn || !navLinks) return;

    menuBtn.addEventListener('click', () => {
        const isOpen = navLinks.style.display === 'flex';
        navLinks.style.display = isOpen ? 'none' : 'flex';
        navLinks.style.flexDirection = 'column';
        navLinks.style.position = 'absolute';
        navLinks.style.top = '64px';
        navLinks.style.left = '0';
        navLinks.style.right = '0';
        navLinks.style.background = 'rgba(10,10,15,0.98)';
        navLinks.style.padding = '1rem 1.5rem';
        navLinks.style.borderBottom = '1px solid #2a2a3a';
        navLinks.style.zIndex = '99';
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
        if (!menuBtn.contains(e.target) && !navLinks.contains(e.target)) {
            if (window.innerWidth <= 768) {
                navLinks.style.display = 'none';
            }
        }
    });
}

/**
 * Subtle scroll-triggered fade-in for cards
 */
function initScrollAnimations() {
    const cards = document.querySelectorAll(
        '.upload-card, .result-image-card, .result-prediction-card, ' +
        '.about-card, .step-card, .class-full-item, .metric-card'
    );

    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    cards.forEach((card, idx) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(16px)';
        card.style.transition = `opacity 0.5s ease ${idx * 0.05}s, transform 0.5s ease ${idx * 0.05}s`;
        observer.observe(card);
    });
}

/**
 * Extra visual feedback for drag-over
 */
function initDropzoneHighlight() {
    const dropzone = document.getElementById('dropzone');
    if (!dropzone) return;

    // Prevent browser default file open on drag
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, (e) => e.preventDefault(), false);
    });
}
