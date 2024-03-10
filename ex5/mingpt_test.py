import torch
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate(model, prompt='', num_samples=10, steps=20, do_sample=True):
        
    # tokenize the input prompt into integer input sequence
    tokenizer = BPETokenizer()
    if prompt == '':
        # to create unconditional samples...
        # manually create a tensor with only the special <|endoftext|> token
        # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
        x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    else:
        x = tokenizer(prompt).to(device)
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)
    
def main():
    torch.manual_seed(42)

    model_type = 'gpt2'
    model = GPT.from_pretrained(model_type)

    # ship model to device and set to eval mode
    model.to(device)
    model.eval()

    print(generate(model, prompt='Life means', num_samples=10, steps=20))

    # generate some samples
    # for prefix in ['Life means',
    #                'Machine learning',
    #                'My name',
    #                'Rock is',
    #                'Love requires',
    #                'Forgiveness is',
    #                'Space travel',
    #                '']:
    #     generate(model, num_samples=5, steps=20)

if __name__ == '__main__':
    main()