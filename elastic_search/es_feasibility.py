import elasticsearch
import numpy as np
import sys
import os
import tqdm
import time

def get_result(es, query, verbose=True):
	num_results = 10
	if not es:
		es = elasticsearch.Elasticsearch()  # use default of localhost, port 9200

	results = es.search(
				    index='cbt',
				    body={
				        "size": num_results,
				        "query": {"match": {"text": {"query": query,"operator": "and"}}}})
	if verbose:
		print "Got %d hits: "%results['hits']['total']
		for hit in results['hits']['hits']:
			print hit
def add_data(file_path):
	es = elasticsearch.Elasticsearch()  # use default of localhost, port 9200
	data = open(file_path)
	line_count = 0
	# limit = 100000
	for line in tqdm.tqdm(data.readlines()):
		line_count += 1
		if line_count % 5000 == 0:
			print line
			print line_count
		es.index(index='cbt', doc_type='raw_text', id=line_count, body = {'text': line})

	# close file
	data.close()
	print "Finished saving data"
	get_result(es,'regard affection')

def query_interactive():
	es = elasticsearch.Elasticsearch()  # use default of localhost, port 9200
	while True:
		query = raw_input("input: ")
		if query == '':
			break
		get_result(es, query)
def benchmark(file_path):
	es = elasticsearch.Elasticsearch()  # use default of localhost, port 9200
	start_time = time.time()
	count = 0
	with open(file_path) as f:
		for line in f.readlines():
			splitted = line.split("\t")
			query = " ".join(splitted[1:3])
			get_result(es, query, verbose=False)
			count += 1
		process_time = time.time() - start_time
	print "Searched for %d in %f seconds (%f search/second)" %(count, process_time, float(count) / process_time)

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print "Specify a mode: inter, index, benchmark"
		sys.exit()

	mode = sys.argv[1]

	if mode == "inter":
		query_interactive()
	elif mode == "index":
		file_path = sys.argv[2] # /data/lisa/exp/jastrzes/l2lwe/data/CBTest/data/cbt_train.txt in case you're looking for it :D
		add_data(file_path=file_path)
	elif mode =="benchmark":
		print "Make sure you've already indexed the data"
		file_path = sys.argv[2] # path/to/train100k.txt
		benchmark(file_path)
