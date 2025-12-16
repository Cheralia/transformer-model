import torch
import math
from tokenizers import Tokenizer
from architecture.architecture import DisentangledTransformer, ModelConfig

DEVICE = 'cpu'

def generate_text(model, tokenizer, start_text, max_new_tokens=100):
    model.eval()
    ids = tokenizer.encode(start_text).ids
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_new_tokens):
        # Crop context if too long
        idx_cond = idx if idx.size(1) <= 128 else idx[:, -128:]
        logits, _ = model(idx_cond)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return tokenizer.decode(idx[0].tolist())

def main():
    print("Loading test data and model...")
    test_data = torch.load("data/test.pt")
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    
    # Init config and load weights
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128)
    model = DisentangledTransformer(config).to(DEVICE)
    model.load_state_dict(torch.load("model.pth"))
    
    # 1. Calculate Test Perplexity
    model.eval()
    # Create a batch from test data (simplified)
    ix = torch.randint(len(test_data) - 128, (32,))
    x = torch.stack([test_data[i:i+128] for i in ix]).to(DEVICE)
    y = torch.stack([test_data[i+1:i+128+1] for i in ix]).to(DEVICE)
    
    with torch.no_grad():
        _, loss = model(x, y)
        ppl = math.exp(loss.item())
    
    print(f"\nTest Set Results:\nLoss: {loss.item():.4f}\nPerplexity: {ppl:.2f}")
    
    # 2. Generation
    print("\n--- Generating Text from 'The King' ---")
    generated = generate_text(model, tokenizer, "The King")
    print(generated)

if __name__ == "__main__":
    main()