import torch

def get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    device = get_torch_device()

    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def get_prediction(model, tokenizer, vocab, prompt):
    device = get_torch_device()
    max_seq_len = 30
    seed = 0
    
    return generate(prompt, max_seq_len, 0.5, model, tokenizer, vocab, device, seed)