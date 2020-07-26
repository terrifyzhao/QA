import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

input_file_name = 'word2vec/wiki.txt'
model_file_name = 'word2vec/wiki.model'

model = Word2Vec(LineSentence(input_file_name),
                 size=100,
                 window=5,
                 sg=1,
                 negative=5,
                 min_count=5,
                 workers=multiprocessing.cpu_count())

model.save(model_file_name)
