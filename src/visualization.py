import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_metrics(metrics):
    """Create training metrics visualization"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss", "Learning Rate"))
    
    fig.add_trace(
        go.Scatter(
            y=[metrics["loss"]],
            mode='lines+markers',
            name='Loss'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=[metrics["learning_rate"]],
            mode='lines+markers',
            name='Learning Rate'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Training Progress"
    )
    
    return fig
