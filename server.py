from flask import Flask, request
from main2 import qa

app = Flask(__name__)


@app.route('/qa')
def question_query():
    text = request.args.get('text')
    answer = qa(text)
    return answer


if __name__ == '__main__':
    app.run()
