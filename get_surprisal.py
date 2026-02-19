import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import Union, Optional, Tuple


def load_surprisal_model(
    model_name: str,
    device: Union[str, torch.device] = None
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a model and tokenizer once for reuse with get_surprisal(..., model=..., tokenizer=...).
    Use this when calling get_surprisal many times with the same model to avoid reloading.

    Returns:
        (model, tokenizer) ready to pass to get_surprisal(prefix=..., word=..., model=model, tokenizer=tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(
                f"Could not load model {model_name}. Supported: CausalLM or MaskedLM. Error: {e}"
            ) from e
    model.to(device)
    model.eval()
    return model, tokenizer


def get_surprisal(
    model_name: Optional[str] = None,
    prefix: Optional[str] = None,
    word: Optional[str] = None,
    base: float = 2.0,
    device: Union[str, torch.device] = None,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None
) -> float:
    """
    Calculate the surprisal of a word given a prefix/context using a Hugging Face model.
    
    Surprisal is defined as: -log_base(P(word|prefix))
    
    Important note on spaces:
    - The function tokenizes prefix+word together to ensure correct tokenization in context
    - This handles cases where tokenizers (especially GPT-2 style) add leading spaces
      when tokenizing words in isolation vs. in context
    - You don't need to worry about adding/removing spaces manually - the function handles it
    
    Args:
        model_name: Name or path of the Hugging Face model (e.g., 'gpt2', 'bert-base-uncased').
                    Required if model and tokenizer are not provided.
        prefix: The context/prefix text before the word. Can include or exclude trailing space.
        word: The word to calculate surprisal for. Can include or exclude leading space.
        base: Base of the logarithm (default: 2.0 for bits, use np.e for nats)
        device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
        model: Pre-loaded model (optional, for efficiency when computing multiple surprisals)
        tokenizer: Pre-loaded tokenizer (optional, for efficiency when computing multiple surprisals).
                   Use load_surprisal_model(model_name) once, then pass the pair to each call.

    Returns:
        The surprisal value (float)
    
    Example:
        >>> # One-off call (loads model each time):
        >>> get_surprisal(model_name='gpt2', prefix='The cat sat on the', word='mat')
        4.23
        
        >>> # Many calls with same model (load once, reuse):
        >>> model, tokenizer = load_surprisal_model('gpt2')
        >>> get_surprisal(prefix='The cat sat on the', word='mat', model=model, tokenizer=tokenizer)
        4.23
        >>> get_surprisal(prefix='The dog', word='ran.', model=model, tokenizer=tokenizer)
        3.12
    """
    # Validate inputs
    if prefix is None or word is None:
        raise ValueError("prefix and word must be provided")
    
    if model is None or tokenizer is None:
        if model_name is None:
            raise ValueError("Either model_name or both model and tokenizer must be provided")
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine model type and load appropriate model class
        try:
            # Try causal LM first (GPT-style models)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_type = 'causal'
        except:
            try:
                # Try masked LM (BERT-style models)
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                model_type = 'masked'
            except Exception as e:
                raise ValueError(f"Could not load model {model_name}. Supported types: CausalLM or MaskedLM. Error: {e}")
        
        model.to(device)
        model.eval()
    else:
        # Use provided model and tokenizer
        if device is None:
            # Try to infer device from model
            device = next(model.parameters()).device
        else:
            device = torch.device(device)
            model.to(device)
        
        # Determine model type (handle concrete classes like GPT2LMHeadModel, not just Auto*)
        model_class_name = type(model).__name__
        if isinstance(model, AutoModelForCausalLM) or 'CausalLM' in model_class_name or 'LMHeadModel' in model_class_name:
            model_type = 'causal'
        elif isinstance(model, AutoModelForMaskedLM) or 'MaskedLM' in model_class_name:
            model_type = 'masked'
        else:
            raise ValueError(
                f"Could not determine model type from {model_class_name}. "
                "Please use AutoModelForCausalLM or AutoModelForMaskedLM (or a concrete causal/masked LM class)."
            )
    
    # Tokenize prefix and word
    # IMPORTANT: Tokenize them together first to get correct tokenization in context
    # This handles space issues (e.g., GPT-2 adds leading spaces when tokenizing words alone)
    # Ensure there's a space between prefix and word for correct tokenization
    # (tokenizers expect word boundaries to have spaces). Empty prefix is not supported (model needs context).
    if not prefix:
        raise ValueError("prefix must not be empty; surprisal requires at least one token of context")
    elif prefix and not prefix.endswith(' ') and word and not word.startswith(' '):
        full_text = prefix + ' ' + word
    else:
        full_text = prefix + word
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt').to(device)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(device)
    
    # Extract word tokens from the full tokenization
    # This ensures word tokens match what the model sees in context
    prefix_len = prefix_tokens.shape[1]
    
    # Verify that prefix tokens match the beginning of full tokens
    # (BPE tokenization should be prefix-stable, but we verify to be safe)
    if prefix_len > 0:
        prefix_in_full = full_tokens[0, :prefix_len]
        if not torch.equal(prefix_tokens[0], prefix_in_full):
            # If they don't match, try to find the prefix alignment
            # This can happen in rare cases with certain tokenizers
            prefix_list = prefix_tokens[0].tolist()
            full_list = full_tokens[0].tolist()
            # Check if prefix tokens appear at the start of full tokens
            if full_list[:prefix_len] != prefix_list:
                raise ValueError(
                    f"Prefix tokenization doesn't align with full text tokenization. "
                    f"This may indicate an issue with space handling. "
                    f"Prefix tokens: {prefix_list}, Full tokens start: {full_list[:prefix_len+3]}"
                )
    
    word_tokens = full_tokens[:, prefix_len:]
    
    if word_tokens.shape[1] == 0:
        raise ValueError(f"Word '{word}' could not be tokenized (or was already part of prefix tokenization)")
    
    with torch.no_grad():
        if model_type == 'causal':
            # For causal models (GPT-style), predict next tokens
            # Handle multi-token words by processing sequentially
            # Use log-probs for numerical stability (sum log-probs then convert to surprisal)
            log_prob_sum = 0.0
            current_prefix = prefix_tokens.clone()
            
            for i, token_id in enumerate(word_tokens[0]):
                # Get logits for the next token position
                outputs = model(current_prefix)
                logits = outputs.logits
                
                # Get logits for the position right after current prefix
                token_logits = logits[0, -1, :]  # Shape: [vocab_size]
                
                # Log-softmax for stability; sum log-probs, convert to surprisal once
                token_log_probs = torch.log_softmax(token_logits, dim=-1)
                token_log_prob = token_log_probs[token_id].item()
                if not np.isfinite(token_log_prob):
                    token_str = tokenizer.decode([token_id.item()])
                    raise ValueError(
                        f"Model assigned zero probability to token {token_id.item()} ({repr(token_str)}) "
                        f"at position {i + 1} of word (prefix + word = {repr(prefix + word)})"
                    )
                log_prob_sum += token_log_prob
                
                # Update prefix for next iteration (add the current token)
                current_prefix = torch.cat([current_prefix, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Surprisal = -log_base(P) = -log(P) / log(base)
            surprisal = -log_prob_sum / np.log(base)
        
        else:  # masked model
            # For masked models (BERT-style), we need to mask the word position
            # Create input with [MASK] token where the word should be
            mask_token = tokenizer.mask_token
            if mask_token is None:
                model_id = model_name if model_name else "provided model"
                raise ValueError(f"Model {model_id} does not have a mask token")
            
            # Tokenize prefix + [MASK]
            # Add a space before mask token if prefix doesn't end with space
            if prefix and not prefix.endswith(' '):
                masked_text = prefix + ' ' + mask_token
            else:
                raise ValueError(f"Prefix must be provided and must not end with a space. Prefix: {prefix}")
            masked_tokens = tokenizer.encode(masked_text, add_special_tokens=True, return_tensors='pt').to(device)
            
            # Find the mask token position
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (masked_tokens == mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_positions) == 0:
                raise ValueError("Could not find mask token position")
            
            # Get logits for masked position
            outputs = model(masked_tokens)
            logits = outputs.logits
            
            # Handle multi-token words (simplified - assumes single token for masked models)
            if word_tokens.shape[1] > 1:
                # For multi-token words with masked models, we need a different approach
                # This is a simplified version - you may need to adjust based on your use case
                word_token_id = word_tokens[0, 0].item()  # Take first token
            else:
                word_token_id = word_tokens[0, 0].item()
            
            # Get log-probability at the mask position (log_softmax for stability)
            mask_pos = mask_positions[0].item()
            token_logits = logits[0, mask_pos, :]
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            token_log_prob = token_log_probs[word_token_id].item()
            if not np.isfinite(token_log_prob):
                token_str = tokenizer.decode([word_token_id])
                raise ValueError(
                    f"Model assigned zero probability to token {word_token_id} ({repr(token_str)}) "
                    f"at mask position (prefix + [MASK] = {repr(masked_text)})"
                )
            surprisal = -token_log_prob / np.log(base)
    
    return surprisal


if __name__ == "__main__":
    # Example usage
    print("Testing surprisal calculation...")
    
    # Test with GPT-2 (causal model)
    print("\nGPT-2 example:")
    surprisal = get_surprisal(model_name='gpt2', prefix='The dog chased the', word='cat.')
    print(f"Surprisal of 'cat.' given 'The dog chased the': {surprisal:.4f} bits")
    surprisal = get_surprisal(model_name='gpt2', prefix='The dog chased the', word='of.')
    print(f"Surprisal of 'of.' given 'The dog chased the ': {surprisal:.4f} bits")
    surprisal = get_surprisal(model_name='gpt2', prefix='The dog', word='barked.')
    print(f"Surprisal of 'barked.' given 'The dog': {surprisal:.4f} bits")