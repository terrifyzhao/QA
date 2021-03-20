from transformers import BertModel, BertTokenizerFast

path = './bert'
tokenizer = BertTokenizerFast.from_pretrained(path)
model = BertModel.from_pretrained(path)

encoding = tokenizer('我爱自然语言处理', return_tensors='pt')
out = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
print(out[1])


from flask import Flask, request
from main2 import qa

app = Flask(__name__)


@app.route('/qa')
def chat():
    text = request.args.get('text')
    res = qa(text)
    return res


if __name__ == '__main__':
    app.run()

