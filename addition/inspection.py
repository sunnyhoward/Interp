import torch as t
from torch import nn
from models import *
from typing import List, Tuple, Dict, Any

def inspect_weights(model: nn.Module):
    """
    Displays the name, shape, and device for all learned parameters (weights and biases)
    in the model.
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name:<30} | Shape: {str(list(param.data.shape)):<20} | Device: {param.device}")


def extract_activations(model: nn.Module, input_data: t.Tensor):
    """
    Extracts the output activations of specific layers using PyTorch forward hooks.
    
    NOTE: The positional encoding is added in the forward method of the model, 
    so we use a forward pre-hook on the first attention layer to capture the
    combined embedding + positional encoding vector.
    """
    activation_storage = {}
    hooks = []
    
    # --- New Pre-Hook for Initial Residual Stream ---
    def pre_hook_fn_initial_input(module, input):
        # The input to the AttentionBlock is a tuple containing the tensor [0]. 
        # This is the combined Embedding + Positional Encoding vector.
        activation_storage['initial_residual_stream'] = input[0].detach()

    # --- Standard Hook for Output Activations ---
    def hook_fn(module, input, output):
        # Check if the output is a tuple (like from AttentionBlock) and select the activation (the first element)
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        
        # Store the detached activation tensor
        activation_storage[module.__name__] = activation.detach()

    # Register hooks on layers we are interested in
    # We remove the hook on model.embed as it is too early.
    target_layers = {
        'layers.0': model.layers[0],
        'layers.1': model.layers[1] if len(model.layers) > 1 else None,
        'out': model.out, # Final hidden state
    }
    
    # Rename modules for easier storage
    for i, layer in enumerate(model.layers):
        layer.__name__ = f'attention_layer_{i}'
    model.out.__name__ = 'final_hidden_state'


    # Register the special PRE-hook for the initial input
    hooks.append(model.layers[0].register_forward_pre_hook(pre_hook_fn_initial_input))


    # Register standard hooks for outputs
    for name, module in target_layers.items():
        if module:
            hooks.append(module.register_forward_hook(hook_fn))

    try:
        # Run a forward pass to trigger the hooks (we discard the result)
        _ = model(input_data)
    except Exception as e:
        print(f"Error during forward pass with hooks: {e}")
    finally:
        # Crucially, remove all hooks after use
        for hook in hooks:
            hook.remove()
            
    return activation_storage


def calculate_manual_attention(model: StackedAttentionModel, input_data: t.Tensor, layer_index: int = 0) -> t.Tensor:
    """
    Manually calculates the attention weights for a specific layer and head 
    using the Q, K, and V projection matrices.
    """

    # 1. Get the pre-attention input (Embeddings + Positional Encoding)
    seq_len = input_data.size(1)
    positions = t.arange(seq_len, device=input_data.device).unsqueeze(0)
    x = model.embed(input_data) + model.pos_embed(positions)
    # x shape: (Batch, Seq_len, D_model) -> (2, 8, 32)
    
    # 2. Get the MultiheadAttention module and parameters for the target layer
    attention_block = model.layers[layer_index]
    attn_module = attention_block.attn
    
    d_model = attn_module.embed_dim
    nhead = attn_module.num_heads
    d_head = d_model // nhead
    
    # 3. Extract W_Q, W_K, W_V matrices and corresponding biases (They are concatenated in PyTorch)
    # The 'in_proj' weight/bias contains W_Q, W_K, W_V concatenated along the first dimension.
    W_qkv = attn_module.in_proj_weight
    b_qkv = attn_module.in_proj_bias

    # 4. Apply the projection and split into Q, K, V
    # x @ W_qkv.T + b_qkv
    qkv_projection = t.einsum("bsd,od->bso", x, W_qkv) + b_qkv
    # qkv_projection shape: (Batch, Seq_len, 3 * D_model) -> (2, 8, 96)
    
    # Split the result into Q, K, and V tensors
    Q, K, V = qkv_projection.chunk(3, dim=-1)
    # Q, K, V shape: (Batch, Seq_len, D_model) -> (2, 8, 32)

    # 5. Split Q, K, V into N_head chunks (Multi-head split)
    # Rearrange to (Batch, N_head, Seq_len, D_head)
    def split_heads(tensor):
        return tensor.view(
            -1, seq_len, nhead, d_head
        ).transpose(1, 2)
    
    Q_h, K_h, V_h = split_heads(Q), split_heads(K), split_heads(V)
    # Q_h, K_h, V_h shape: (Batch, N_head, Seq_len, D_head) -> (2, 4, 8, 8)

    # 6. Calculate Attention Scores (QK^T) across all heads simultaneously
    # attn_scores shape: (Batch, N_head, Seq_len_Q, Seq_len_K) -> (2, 4, 8, 8)
    attn_scores = t.matmul(Q_h, K_h.transpose(-2, -1))

    # 7. Apply Scaling Factor (1 / sqrt(d_head))
    scaling_factor = d_head ** 0.5
    attn_scores_scaled = attn_scores / scaling_factor

    # 8. Apply Softmax to get Attention Weights (Probability distribution)
    attn_weights_manual = t.softmax(attn_scores_scaled, dim=-1)
    
    # print(f"Manual Attention Weights (Scores) calculated for Batch 0, Head {head_index}. Shape: {list(attn_weights_manual.shape)}")
    # print(f"First row of weights (Token 0 attending to all tokens):\n{attn_weights_manual[0, 0, :].cpu().numpy()}")
    
    return attn_weights_manual

def extract_attention_matrices(model: nn.Module, layer_index: int = 0) -> Dict[str, t.Tensor]:
    """
    Extracts the individual Query, Key, and Value (W and b) projection matrices 
    for a specific attention layer.
    """
    
    # 1. Access the MultiheadAttention module
    # Check if the model has the 'layers' attribute (i.e., it's StackedAttentionModel)
    if not hasattr(model, 'layers') or len(model.layers) <= layer_index:
        raise IndexError(f"Layer index {layer_index} is out of bounds or model structure is incorrect.")
        
    attention_block = model.layers[layer_index]
    attn_module = attention_block.attn
    
    # 2. Extract concatenated weights and biases
    # W_qkv shape: (3 * D_model, D_model)
    W_qkv = attn_module.in_proj_weight
    # b_qkv shape: (3 * D_model,)
    b_qkv = attn_module.in_proj_bias
    
    # 3. Split the concatenated tensors along dimension 0 (the output dimension)
    W_q, W_k, W_v = W_qkv.chunk(3, dim=0)
    b_q, b_k, b_v = b_qkv.chunk(3, dim=0)

    print(f"W_Q shape: {list(W_q.shape)}")
    print(f"W_K shape: {list(W_k.shape)}")
    print(f"W_V shape: {list(W_v.shape)}")
    print(f"b_Q shape: {list(b_q.shape)}")
    
    return {
        "W_Q": W_q, "b_Q": b_q,
        "W_K": W_k, "b_K": b_k,
        "W_V": W_v, "b_V": b_v,
    }


def extract_io_matrices(model: nn.Module) -> Dict[str, t.Tensor]:
    """
    Extracts the static weights for the input (Embedding/Positional) and 
    output (Linear) layers.
    """
    
    if not hasattr(model, 'embed') or not hasattr(model, 'out'):
        raise AttributeError("Model must have 'embed' and 'out' attributes for this function.")
        
    W_embed = model.embed.weight
    W_pos = model.pos_embed.weight
    W_out = model.out.weight
    b_out = model.out.bias

    print(f"W_Embed shape: {list(W_embed.shape)} (Vocab Size x D_model)")
    print(f"W_Pos shape: {list(W_pos.shape)} (Max Seq Len x D_model)")
    print(f"W_Out shape: {list(W_out.shape)} (Vocab Size x D_model)")
    print(f"b_Out shape: {list(b_out.shape)}")
    
    return {
        "W_Embed": W_embed, 
        "W_Pos": W_pos, 
        "W_Out": W_out, 
        "b_Out": b_out,
    }

"""
Evaluation metrics

"""

def test_performance_by_input_format(model, dataset):
    """Test accuracy on different addition formats"""
    
    # Generate test cases for each format
    formats = {
        'double+double': [],      # a1 a2 + b1 b2 = (e.g., 23+47)
        'double+triple': [],      # a1 a2 + b1 b2 b3 = (e.g., 23+478)
        'triple+double': [],       # a1 a2 a3 + b1 b2 = (e.g., 345+67)
        'triple+triple': []       # a1 a2 a3 + b1 b2 b3 = (e.g., 345+678)
    }
    
    # Categorize test examples by format
    for a, b in dataset.test_examples:
        a_digits = len(str(a))
        b_digits = len(str(b))

        if a_digits == 2 and b_digits == 2:
            formats['double+double'].append((a, b))
        elif a_digits == 2 and b_digits == 3:
            formats['double+triple'].append((a, b))
        elif a_digits == 3 and b_digits == 2:
            formats['triple+double'].append((a, b))
        elif a_digits == 3 and b_digits == 3:
            formats['triple+triple'].append((a, b))   

    
    print(f"{'Format':<20} {'Count':<8} {'Correct':<10} {'Accuracy':<10}")
    print("="*50)
    
    results = {}
    
    for format_name, test_cases in formats.items():
        if len(test_cases) == 0:
            continue
            
        correct = 0
        total = len(test_cases)
        
        for a, b in test_cases:
            inp_tensor, out_tensor = dataset.get_example(a, b)
            
            with t.no_grad():
                logits = model(inp_tensor)
                pred = logits[0, -dataset.max_output_len:].argmax(dim=-1)
                
                if (pred == out_tensor[0, -dataset.max_output_len:]).all():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[format_name] = {'correct': correct, 'total': total, 'accuracy': accuracy}
        
        print(f"{format_name:<20} {total:<8} {correct:<10} {accuracy*100:>6.2f}%")
    
    print("\n" + "="*50)
    
    # Overall accuracy
    total_correct = sum(r['correct'] for r in results.values())
    total_count = sum(r['total'] for r in results.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"{'Overall':<20} {total_count:<8} {total_correct:<10} {overall_acc*100:>6.2f}%")

    
    return results



def test_performance_by_output_format(model, dataset):
    """Test accuracy based on the number of digits in the output sum"""
    
    # Generate test cases for each output format
    formats = {
        'two_digit_sum': [],      # e.g., 23 + 45 = 68
        'three_digit_sum': [],    # e.g., 56 + 78 = 134
        'four_digit_sum': []      # e.g., 345 + 678 = 1023
    }
    
    # Categorize test examples by output format
    for a, b in dataset.test_examples:
        total = a + b
        total_digits = len(str(total))

        if total_digits == 2:
            formats['two_digit_sum'].append((a, b))
        elif total_digits == 3:
            formats['three_digit_sum'].append((a, b))
        elif total_digits == 4:
            formats['four_digit_sum'].append((a, b))   
    
    print(f"{'Output Format':<20} {'Count':<8} {'Correct':<10} {'Accuracy':<10}")
    print("="*50)
    
    results = {}
    
    for format_name, test_cases in formats.items():
        if len(test_cases) == 0:
            continue
            
        correct = 0
        total = len(test_cases)
        
        for a, b in test_cases:
            inp_tensor, out_tensor = dataset.get_example(a,b)
            
            with t.no_grad():
                logits = model(inp_tensor)
                pred = logits[0, -dataset.max_output_len:].argmax(dim=-1)
                
                if (pred == out_tensor[0, -dataset.max_output_len:]).all():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[format_name] = {'correct': correct, 'total': total, 'accuracy': accuracy}
        
        print(f"{format_name:<20} {total:<8} {correct:<10} {accuracy*100:>6.2f}%")