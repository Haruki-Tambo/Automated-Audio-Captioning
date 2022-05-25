from gensim.models.word2vec import LineSentence, Word2Vec
output_file = 'data/target_development_pad.txt'
sentences = LineSentence(output_file)
model = Word2Vec(sentences, vector_size=192, min_count=1, workers=8)
model.train(sentences, total_examples=model.corpus_count, epochs=1000)
model.save("data/w2v_192_pad.mod")
print("Done")