# 🏎️ DQN CarRacing AI

[![Deploy to Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)

> **Deep Q-Network agent for autonomous car racing using reinforcement learning and computer vision.**

## 🎯 Project Overview

Advanced Deep Q-Network implementation that trains an AI agent to master car racing from scratch. Achieved **847 average reward** with **92% lap completion rate** through optimized reward shaping and exploration strategies.

## 🚀 Live Demo

**:** [DEMO]([https://dqn-carracing-portfolio.onrender.com](https://dqn-carracing-rl-x9h7.onrender.com))

## ⚡ Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/vanshtaneja23/dqn-carracing.git
cd dqn-carracing

# Install dependencies
pip install -r requirements.txt

# Run web portfolio
python app.py
```

Visit `http://localhost:8080` to view the portfolio.



## 🏆 Key Results

| Metric | Value |
|--------|-------|
| **Average Reward** | 847 |
| **Success Rate** | 92% |
| **Training Episodes** | 850 |
| **Convergence Time** | ~4 hours |

## 🛠️ Technical Implementation

### Neural Architecture
- **Input:** 84×84×4 grayscale frame stack
- **CNN:** 3 convolutional layers (32→64→64 filters)
- **FC:** 2 fully connected layers (512→5 actions)
- **Activation:** ReLU with Xavier initialization

### Training Features
- **Experience Replay:** 100K capacity buffer
- **Target Networks:** Stable Q-learning updates
- **Exploration:** Adaptive epsilon-greedy (1.0→0.01)
- **Reward Shaping:** Custom rewards for racing optimization

### Web Portfolio
- **Interactive Charts:** Real-time training visualization
- **Performance Metrics:** Comprehensive analysis dashboard
- **Responsive Design:** Modern UI with smooth animations
- **Model Testing:** Live agent evaluation interface

## 📁 Project Structure

```
├── app.py                 # Flask web application
├── dqn_agent.py          # DQN implementation
├── environment.py        # CarRacing environment wrapper
├── training.py           # Training pipeline
├── analysis.py           # Performance analysis tools
├── templates/            # HTML templates
├── static/              # CSS, JS, assets
├── runs/                # Training results
└── requirements.txt     # Dependencies
```

## 🔧 Technologies

- **Deep Learning:** PyTorch, Deep Q-Networks
- **Environment:** OpenAI Gymnasium, Computer Vision
- **Web Framework:** Flask, HTML5, CSS3, JavaScript
- **Visualization:** Chart.js, Matplotlib, Seaborn
- **Deployment:** Render, Gunicorn

## 📊 Training Process

1. **State Preprocessing:** Frame stacking and grayscale conversion
2. **Action Selection:** Epsilon-greedy exploration with decay
3. **Experience Storage:** Replay buffer for stable learning
4. **Network Updates:** Target network synchronization
5. **Performance Tracking:** Real-time metrics and diagnostics

## 🎨 Portfolio Features

- **Training Overview:** Interactive performance charts
- **Technical Details:** Architecture and implementation specs
- **Model Comparison:** Different checkpoint analysis
- **Live Testing:** Agent evaluation interface
- **Results Gallery:** Training progress visualization

## 📈 Performance Analysis

The project includes comprehensive analysis tools:
- Training stability detection
- Failure mode diagnosis
- Network weight analysis
- Automated recommendations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the CarRacing environment
- PyTorch team for the deep learning framework
- Render for deployment platform

---

**Built with ❤️ for the ML community**

*Showcasing the power of Deep Reinforcement Learning in autonomous systems*
