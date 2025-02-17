import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_metrics(metrics):
    """Create training metrics visualization"""
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Training Progress"))

    fig.add_trace(
        go.Scatter(
            y=[metrics["loss"]],
            mode='lines+markers',
            name='Loss'
        )
    )

    fig.update_layout(
        height=400,
        showlegend=True,
        title_text=f"Training Progress (Epoch {metrics['epoch']})",
        xaxis_title="Steps",
        yaxis_title="Loss"
    )

    return fig

def plot_training_history(metrics_history):
    """Plot complete training history"""
    epochs = [m["epoch"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode='lines+markers',
            name='Training Loss'
        )
    )

    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        showlegend=True
    )

    return fig