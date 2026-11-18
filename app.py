#!/usr/bin/env python3
"""
Professional DQN CarRacing Portfolio Web Application
Deploy-ready Flask app for Render
"""

from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'dqn-carracing-portfolio-2024'

# Flask app configuration

class PortfolioData:
    """Manage portfolio data and analytics"""
    
    def __init__(self):
        self.load_project_data()
    
    def load_project_data(self):
        """Load training runs and their real, measured results"""
        self.runs = self.load_training_runs()
        self.project_stats = self.calculate_project_stats()
        self.performance_data = self.generate_performance_data()

    @staticmethod
    def _read_json(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def load_training_runs(self):
        """Load all training runs with their logged rewards and evaluation results"""
        runs_dir = 'runs'
        runs = []

        if os.path.exists(runs_dir):
            for run_name in os.listdir(runs_dir):
                if run_name.startswith('dqn_carracing'):
                    run_path = os.path.join(runs_dir, run_name)
                    config = self._read_json(os.path.join(run_path, 'config.json'))
                    if config is None:
                        continue
                    runs.append({
                        'name': run_name,
                        'path': run_path,
                        'config': config,
                        'timestamp': datetime.fromtimestamp(os.path.getctime(run_path)),
                        'has_progress_image': os.path.exists(os.path.join(run_path, 'training_progress.png')),
                        'episodes': config.get('num_episodes', 0),
                        'rewards': self._read_json(os.path.join(run_path, 'rewards.json')),
                        'eval': self._read_json(os.path.join(run_path, 'eval.json')),
                    })

        return sorted(runs, key=lambda x: x['timestamp'], reverse=True)

    def calculate_project_stats(self):
        """Stats come only from eval.json (written by evaluate_agent.py on the standard reward)"""
        evaluated = [r for r in self.runs if r['eval'] and r['eval'].get('environment') == 'CarRacing-v3']
        best = max(evaluated, key=lambda r: r['eval']['avg_reward']) if evaluated else None
        return {
            'total_runs': len(self.runs),
            'total_episodes': sum(len(r['rewards']['raw']) for r in self.runs if r['rewards']),
            'avg_reward': round(best['eval']['avg_reward']) if best else None,
            'success_rate': round(100 * best['eval']['share_900_plus']) if best else None,
            'eval_episodes': best['eval']['episodes'] if best else None,
            'latest_run': self.runs[0] if self.runs else None,
        }

    def generate_performance_data(self):
        """Learning curve from the latest run's logged raw (unshaped) rewards"""
        logged = [r for r in self.runs if r['rewards'] and r['rewards'].get('raw')]
        if not logged:
            return {}
        rewards = logged[0]['rewards']['raw']
        window = 50
        moving_avg = [float(np.mean(rewards[max(0, i - window + 1):i + 1])) for i in range(len(rewards))]
        return {
            'episodes': list(range(len(rewards))),
            'rewards': rewards,
            'moving_average': moving_avg,
            'final_performance': float(np.mean(rewards[-100:])),
            'peak_performance': float(max(rewards)),
            'learning_stability': float(np.std(rewards[-200:])),
        }

    def create_performance_chart(self):
        """Create performance visualization as base64 image"""
        if not self.performance_data:
            return None
        
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = self.performance_data['episodes']
        rewards = self.performance_data['rewards']
        moving_avg = self.performance_data['moving_average']
        
        # Episode rewards
        ax1.plot(episodes, rewards, alpha=0.3, color='#3498db', linewidth=0.5, label='Episode Rewards')
        ax1.plot(episodes, moving_avg, color='#e74c3c', linewidth=2, label='Moving Average (50 episodes)')
        ax1.set_title('DQN Training Progress - Episode Rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance distribution
        ax2.hist(rewards, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
        ax2.axvline(np.median(rewards), color='#f39c12', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.1f}')
        ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64

# Initialize portfolio data
portfolio = PortfolioData()

@app.route('/')
def index():
    """Main portfolio page"""
    return render_template('index.html', 
                         project_stats=portfolio.project_stats,
                         performance_data=portfolio.performance_data)

@app.route('/api/performance-chart')
def performance_chart():
    """API endpoint for performance chart"""
    chart_data = portfolio.create_performance_chart()
    return jsonify({'chart': chart_data})

@app.route('/api/project-data')
def project_data():
    """API endpoint for project data"""
    return jsonify({
        'stats': portfolio.project_stats,
        'performance': portfolio.performance_data,
        'runs': [
            {
                'name': run['name'],
                'episodes': run['episodes'],
                'timestamp': run['timestamp'].isoformat(),
                'config': run['config'],
                'eval': run['eval']
            }
            for run in portfolio.runs
        ]
    })

@app.route('/training-progress/<run_name>')
def training_progress(run_name):
    """Serve training progress images"""
    run_path = os.path.join('runs', run_name)
    image_path = os.path.join(run_path, 'training_progress.png')
    
    if os.path.exists(image_path):
        return send_from_directory(run_path, 'training_progress.png')
    else:
        return "Image not found", 404

@app.route('/health')
def health():
    """Health check endpoint for deployment"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # For local development and deployment
    import os
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)