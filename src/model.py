import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block with dilated causal convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # Causal padding to maintain temporal causality
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # Apply causal padding
        residual = self.residual(x)
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout1(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout2(out)
        
        # Residual connection
        out = out + residual
        
        return out

class MultiScaleTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network with 16 parallel streams"""
    
    def __init__(self, num_channels, kernel_sizes, dilation_rates, num_blocks=3, dropout=0.2):
        super(MultiScaleTCN, self).__init__()
        
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.num_blocks = num_blocks
        
        # Create 16 parallel TCN streams
        self.tcn_streams = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            stream = nn.ModuleList()
            in_channels = num_channels
            
            for block_idx in range(num_blocks):
                dilation = dilation_rates[block_idx % len(dilation_rates)]
                out_channels = 128  # Fixed output channels for each stream
                
                tcn_block = TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
                stream.append(tcn_block)
                in_channels = out_channels
            
            self.tcn_streams.append(stream)
    
    def forward(self, x):
        # x shape: (batch, channels, time_points)
        
        # Process through each TCN stream
        stream_outputs = []
        
        for stream in self.tcn_streams:
            stream_input = x
            
            # Pass through TCN blocks in the stream
            for tcn_block in stream:
                stream_input = tcn_block(stream_input)
            
            stream_outputs.append(stream_input)
        
        # Concatenate all stream outputs
        # Each stream output: (batch, 128, time_points)
        # Concatenated: (batch, 16*128=2048, time_points)
        concatenated = torch.cat(stream_outputs, dim=1)
        
        return concatenated

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for EEG channel weighting"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x):
        # x shape: (batch, features)
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, self.num_heads, self.d_k)
        K = self.w_k(x).view(batch_size, self.num_heads, self.d_k)
        V = self.w_v(x).view(batch_size, self.num_heads, self.d_k)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.view(batch_size, self.d_model)
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output, attention_weights

class EmoTCNAttn(nn.Module):
    """Main EmoTCN-Attn model: Multi-Scale TCN with Attention for EEG Emotion Recognition"""
    
    def __init__(self, num_channels=62, num_classes=3, kernel_sizes=None, 
                 dilation_rates=None, num_tcn_blocks=3, num_attention_heads=8, 
                 dropout_rate=0.2):
        super(EmoTCNAttn, self).__init__()
        
        # Default parameters as specified in the paper
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
        if dilation_rates is None:
            dilation_rates = [1, 2, 4]
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Multi-Scale TCN
        self.multi_scale_tcn = MultiScaleTCN(
            num_channels=num_channels,
            kernel_sizes=kernel_sizes,
            dilation_rates=dilation_rates,
            num_blocks=num_tcn_blocks,
            dropout=dropout_rate
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature dimension after concatenation
        self.feature_dim = len(kernel_sizes) * 128  # 16 streams * 128 channels = 2048
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            d_model=self.feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time_points)
        
        # Multi-Scale TCN processing
        tcn_output = self.multi_scale_tcn(x)  # (batch, 2048, time_points)
        
        # Global Average Pooling
        pooled = self.global_avg_pool(tcn_output).squeeze(-1)  # (batch, 2048)
        
        # Multi-Head Attention
        attended, attention_weights = self.attention(pooled)  # (batch, 2048)
        
        # Classification
        logits = self.classifier(attended)  # (batch, num_classes)
        
        return logits, attention_weights

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    batch_size = 4
    num_channels = 62
    time_points = 800  # 4 seconds * 200 Hz
    
    # Create model
    model = EmoTCNAttn(num_classes=3)  # SEED dataset
    
    # Create dummy input
    x = torch.randn(batch_size, num_channels, time_points)
    
    # Forward pass
    logits, attention_weights = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Total parameters: {count_parameters(model):,}") 