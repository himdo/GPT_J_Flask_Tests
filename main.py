import json
from transformers import GPTJForCausalLM
import torch
from flask import Flask
from flask import request
app = Flask(__name__)
#Will need at least 13-14GB of Vram for CUDA
if torch.cuda.is_available():
    print('GPU')
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).cuda()
else:
    print('CPU')
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model.eval()
print('Ready for Inputs')
def GPTJ_GetResponse(data):
    input_text = data['prompt']
    input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

    output = model.generate(
        input_ids,
        do_sample=data['do_sample'],
        max_length=data['max_length'],
        top_p=data['top_p'],
        top_k=data['top_k'],
        temperature=data['temperature'],
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.route("/", methods = ['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        data = request.data

        print(json.loads(data))
        return GPTJ_GetResponse(json.loads(data))
    return "<p>Hello, World!</p>"