import collections
import numpy as np
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
								'090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
	with open(d2l.download('time_machine'), 'r') as f:
		lines = f.readlines()
	return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
	if token == 'word':
		return [line.split() for line in lines]
	elif token == 'char':
		return [list(line) for line in lines]
	else:
		print('invalid token type: ' + token)

def count_corpus(tokens):
	if len(tokens) == 0 or isinstance(tokens[0], list):
		tokens = [token for line in tokens for token in line]
	return collections.Counter(tokens)

def combine_corpus(tokens):
	return '-'.join(tokens)

def count_combined_corpus(tokens, n=2):
	seq = []
	combined_corpus_freq = dict()
	for token in tokens:
		seq.append(token)
		if len(seq) == n:
			combined_corpus = combine_corpus(seq)
			combined_corpus_freq[combined_corpus] = (
				combined_corpus_freq.get(combined_corpus, 0) + 1)
			seq.pop(0)
	return combined_corpus_freq


class Vocab:
	def __init__(self, tokens=None, min_freq=0, n=2):
		if tokens is None:
			tokens = []
		self.token_freqs = count_corpus(tokens)
		sorted_freqs = sorted(self.token_freqs.items(), key=lambda x: x[1],
							  reverse=True)
		self.unk, uniq_tokens = 0, ['<unk>'] # + reserved_tokens
		uniq_tokens += [token for token, freq in sorted_freqs
						if freq >= min_freq and token not in uniq_tokens]
		self.idx_to_token, self.token_to_idx = [], dict()
		for token in uniq_tokens:
			self.idx_to_token.append(token)
			self.token_to_idx[token] = len(self.idx_to_token) - 1
		self.combined_corpus_freq = count_combined_corpus(tokens)
		self.token_size = len(tokens)

	def __len__(self):
		return len(self.idx_to_token)

	def __getitem__(self, tokens):
		if not isinstance(tokens, (list, tuple)):
			return self.token_to_idx.get(tokens, self.unk)
		return [self.__getitem__(token) for token in tokens]

	def get_count(self, tokens):
		if len(tokens) > 1:
			return self.combined_corpus_freq.get(combine_corpus(tokens), 0)
		elif len(tokens) == 1:
			return self.token_freqs.get(tokens[0], 0)
		return 0

	def to_tokens(self, indices):
		if not isinstance(indices, (list, tuple)):
			return self.idx_to_token[indices]
		return [self.idx_to_token[index] for index in indices]

	def corpus_size(self):
		return self.token_size


class Predictor:
	def __init__(self, seq_size=4, t=2, token='char'):
		lines = read_time_machine()
		tokens = sum(tokenize(lines, token=token), [])
		self.vocab = Vocab(tokens)
		self.total = len(tokens)
		self.seq_size = seq_size
		self.t = t
		self.last_tokens = tokens[-(seq_size - 1):]

	# markov chain implementation.
	def probability(self, seq):
		prob = 1
		markov_seq = []
		for token in seq:
			markov_seq.append(token)
			markov_seq = markov_seq if len(markov_seq) <= self.t else markov_seq[-self.t:]
			count = self.vocab.get_count(markov_seq)
			prev_count = self.total if len(markov_seq) == 1 else self.vocab.get_count(markov_seq[:-1])
			# e.g. case p(x3|x1,x2) we simply return 0 if we cannot find n(x1,x2).
			# This can be optimized later.
			if count == 0 or prev_count == 0:
				return 0
			term = float(count) / count
			prob *= term
		return prob

	def next(self):
		best_guess = max([(word, self.probability(self.last_tokens + [word])) for word in self.vocab.idx_to_token],
						 key=lambda x:x[1])[0]
		self.last_tokens = self.last_tokens[1:] + [best_guess]
		return best_guess


predicator = Predictor(seq_size=6, t=2, token='word')
for i in range(20):
	print(predicator.next())