import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    LSTM Decoder for caption generation.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the LSTM decoder.
        
        Args:
            embed_size (int): Size of word embeddings and image features
            hidden_size (int): Size of LSTM hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of LSTM layers
        """
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to produce vocabulary scores
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        Forward pass during training.
        
        Args:
            features (torch.Tensor): Image features from encoder (batch_size, embed_size)
            captions (torch.Tensor): Target captions (batch_size, caption_length)
            
        Returns:
            torch.Tensor: Vocabulary scores (batch_size, caption_length, vocab_size)
        """
        # Remove <end> token from captions and embed them
        cap_embedding = self.embed(captions[:, :-1])
        
        # Concatenate image features with caption embeddings
        # Add image features as the first input to LSTM
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Generate vocabulary scores
        outputs = self.linear(lstm_out)
        
        return outputs
    
    def sample(self, features, states=None, max_len=20):
        """
        Generate captions using greedy search during inference.
        
        Args:
            features (torch.Tensor): Image features (1, embed_size)
            states (tuple): Initial LSTM states
            max_len (int): Maximum caption length
            
        Returns:
            list: Generated word indices
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        for i in range(max_len):
            # Pass through LSTM
            hiddens, states = self.lstm(inputs, states)
            
            # Generate vocabulary scores
            outputs = self.linear(hiddens.squeeze(1))  # (1, vocab_size)
            
            # Get predicted word index
            predicted = outputs.max(1)[1]  # (1,)
            sampled_ids.append(predicted.item())
            
            # Break if <end> token is generated (assuming index 1 is <end>)
            if predicted == 1:
                break
                
            # Prepare input for next iteration
            inputs = self.embed(predicted)  # (1, embed_size)
            inputs = inputs.unsqueeze(1)  # (1, 1, embed_size)
            
        return sampled_ids
