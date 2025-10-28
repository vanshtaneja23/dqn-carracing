// DQN CarRacing Portfolio JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Initialize particles background
    initParticles();

    // Initialize performance chart
    initPerformanceChart();

    // Initialize smooth scrolling
    initSmoothScrolling();

    // Initialize navbar scroll effect
    initNavbarScroll();

    // Initialize animations
    initAnimations();
});

// Particles.js configuration
function initParticles() {
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: {
                    value: 80,
                    density: {
                        enable: true,
                        value_area: 800
                    }
                },
                color: {
                    value: ['#fbbf24', '#f59e0b', '#d97706', '#ffffff']
                },
                shape: {
                    type: 'circle',
                    stroke: {
                        width: 0,
                        color: '#000000'
                    }
                },
                opacity: {
                    value: 0.5,
                    random: false,
                    anim: {
                        enable: false,
                        speed: 1,
                        opacity_min: 0.1,
                        sync: false
                    }
                },
                size: {
                    value: 3,
                    random: true,
                    anim: {
                        enable: false,
                        speed: 40,
                        size_min: 0.1,
                        sync: false
                    }
                },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#fbbf24',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 6,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false,
                    attract: {
                        enable: false,
                        rotateX: 600,
                        rotateY: 1200
                    }
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: {
                        enable: true,
                        mode: 'repulse'
                    },
                    onclick: {
                        enable: true,
                        mode: 'push'
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 400,
                        line_linked: {
                            opacity: 1
                        }
                    },
                    bubble: {
                        distance: 400,
                        size: 40,
                        duration: 2,
                        opacity: 8,
                        speed: 3
                    },
                    repulse: {
                        distance: 200,
                        duration: 0.4
                    },
                    push: {
                        particles_nb: 4
                    },
                    remove: {
                        particles_nb: 2
                    }
                }
            },
            retina_detect: true
        });
    }
}

// Performance Chart
function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx || !window.performanceData) return;

    const data = window.performanceData;

    // Create gradient
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(251, 191, 36, 0.8)');
    gradient.addColorStop(1, 'rgba(251, 191, 36, 0.1)');

    const movingAvgGradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
    movingAvgGradient.addColorStop(0, 'rgba(245, 158, 11, 0.8)');
    movingAvgGradient.addColorStop(1, 'rgba(245, 158, 11, 0.1)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.episodes,
            datasets: [
                {
                    label: 'Episode Rewards',
                    data: data.rewards,
                    borderColor: '#fbbf24',
                    backgroundColor: gradient,
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.1
                },
                {
                    label: 'Moving Average',
                    data: data.moving_average,
                    borderColor: '#f59e0b',
                    backgroundColor: movingAvgGradient,
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: {
                            family: 'Inter',
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#fbbf24',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Episode',
                        color: '#ffffff',
                        font: {
                            family: 'Inter',
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        color: '#ffffff',
                        font: {
                            family: 'Inter'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Reward',
                        color: '#ffffff',
                        font: {
                            family: 'Inter',
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        color: '#ffffff',
                        font: {
                            family: 'Inter'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    // Hide loading indicator
    const loading = document.getElementById('chartLoading');
    if (loading) {
        loading.style.display = 'none';
    }
}

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Navbar scroll effect
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;

    window.addEventListener('scroll', function () {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }

        lastScrollTop = scrollTop;
    });
}

// Initialize animations on scroll
function initAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.overview-card, .metric-card, .tech-card, .achievement-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Counter animation for stats
function animateCounters() {
    const counters = document.querySelectorAll('.stat-number, .metric-value');

    counters.forEach(counter => {
        const target = parseInt(counter.textContent.replace(/[^\d]/g, ''));
        const increment = target / 100;
        let current = 0;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                counter.textContent = target.toLocaleString();
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current).toLocaleString();
            }
        }, 20);
    });
}

// Initialize counter animation when stats section is visible
const statsObserver = new IntersectionObserver(function (entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCounters();
            statsObserver.unobserve(entry.target);
        }
    });
});

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    statsObserver.observe(heroStats);
}

// Add loading states and error handling
window.addEventListener('load', function () {
    // Remove any loading states
    document.querySelectorAll('.loading').forEach(el => {
        el.style.display = 'none';
    });

    // Add loaded class to body for CSS animations
    document.body.classList.add('loaded');
});

// Error handling for chart
window.addEventListener('error', function (e) {
    console.error('Error loading chart:', e);
    const loading = document.getElementById('chartLoading');
    if (loading) {
        loading.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Chart loading failed';
    }
});

// Mobile menu toggle (if needed)
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('active');
}

// Add click handlers for interactive elements
document.addEventListener('click', function (e) {
    // Handle any interactive elements
    if (e.target.matches('.btn')) {
        // Add click animation
        e.target.style.transform = 'scale(0.95)';
        setTimeout(() => {
            e.target.style.transform = '';
        }, 150);
    }
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function () {
        setTimeout(function () {
            const perfData = performance.timing;
            const loadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log('Page load time:', loadTime + 'ms');
        }, 0);
    });
}