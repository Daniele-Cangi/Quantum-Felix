# GitHub Pages Setup for Quantum Felix

This directory contains the GitHub Pages configuration for the Quantum Felix project website.

## 🌐 Website Structure

- **index.md**: Main landing page with project overview
- **features.md**: Detailed feature documentation  
- **documentation.md**: Complete technical documentation
- **examples.md**: Practical examples and tutorials
- **_config.yml**: Jekyll configuration for GitHub Pages
- **Gemfile**: Ruby dependencies for Jekyll

## 🚀 Deployment

The website is automatically deployed using GitHub Actions when changes are pushed to the main/master branch.

### Manual Setup (if needed)

1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"
3. The workflow will automatically build and deploy the site

### Local Development

To run the site locally for testing:

```bash
# Install dependencies
bundle install

# Serve the site locally
bundle exec jekyll serve

# Access at http://localhost:4000
```

## 📄 Page Layout

```
Quantum Felix Website
├── Home (index.md)
│   ├── Project overview
│   ├── Key features highlights
│   ├── Quick start guide
│   └── Navigation to other sections
├── Features (features.md)
│   ├── Scenario Factory
│   ├── Backtesting & Stress Testing
│   ├── Strategy Orchestration
│   ├── Cost & Risk Profiling
│   ├── Quantum Early Stopping
│   └── Integration Capabilities
├── Documentation (documentation.md)
│   ├── Installation guide
│   ├── API reference
│   ├── Architecture overview
│   ├── Configuration management
│   └── Troubleshooting
└── Examples (examples.md)
    ├── Basic examples
    ├── Advanced examples
    ├── Real-world case studies
    └── Learning path
```

## 🎨 Styling

The website uses the Minima theme with custom CSS for:
- Hero sections with gradient backgrounds
- Feature cards with hover effects
- Responsive design for mobile devices
- Professional color scheme
- Interactive buttons and navigation

## 🔧 Configuration

Key configuration options in `_config.yml`:
- Site title and description
- Navigation pages
- Social links
- Plugin configuration
- Build settings

## 📊 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **SEO Optimized**: Proper meta tags and structure
- **Interactive Elements**: Hover effects and smooth transitions
- **Code Highlighting**: Syntax highlighting for code examples
- **Social Integration**: GitHub links and sharing
- **Fast Loading**: Optimized for performance

## 🤝 Contributing

To update the website:
1. Edit the relevant .md files
2. Test locally if possible
3. Commit and push changes
4. GitHub Actions will automatically deploy updates

---

The website showcases Quantum Felix as a professional, cutting-edge simulation engine while providing comprehensive documentation and examples for users.