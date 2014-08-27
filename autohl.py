# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 07:28:50 2014

@author: bmkessle
"""

import nltk
import nltk.data
import numpy as np
from itertools import chain
import argparse

sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
porter = nltk.PorterStemmer()

def read_textfile(filename):
    with open(filename,'r') as f:
        raw_text = f.read()
    return raw_text
    
def process_text_to_sentences(raw_text):
    return sent_tokenizer.tokenize(raw_text.strip())

def process_sentences_to_stems(sentences, min_word_len=5):
    """
    Splits a a list of strings (sentences) into words
    normalizes those to lower case and drops words shorter than min_word length
    it then returns sets of word stems by applying the porter stemmer
    """
    word_sentences = [nltk.wordpunct_tokenize(sentence.decode('utf-8')) for sentence in sentences]
    norm_word_sentences = [[w.lower() for w in sentence if len(w)>=min_word_len] for sentence in word_sentences]
    stem_sentences = [set([porter.stem(w) for w in sentence]) for sentence in norm_word_sentences ]
    return stem_sentences
      
def sim_text_rank(sent_1,sent_2,eps=0.01):
    """
    Similarity between two sets of words given by the number of words in common
    divided by the log of the size of each set.  the parameter eps=0.01 gives the
    score for the pathological case of two identical one word sets
    """
    numerator = len(sent_1.intersection(sent_2))
    if numerator>0:
        return numerator/(np.log(len(sent_1)+eps)+np.log(len(sent_2)+eps))
    return 0
    
def generate_similarity_matrix(stem_sentences,sim_func=sim_text_rank):
    """
    Computes the similarity matrix (symmetric) for a list of sets of words
    using the similarity function sim_func
    """
    N_sentences = len(stem_sentences)
    sim_matrix = np.zeros([N_sentences,N_sentences])
    for i in range(N_sentences):
        for j in range(i+1,N_sentences):
                sim_matrix[i,j] = sim_func(stem_sentences[i],stem_sentences[j])
                sim_matrix[j,i] = sim_matrix[i,j]
    return sim_matrix

def page_rank(sim_matrix,alpha=0.85,iterations=200,tol=1e-8):
    """
    The PageRank algorithm for an undirected weighted graph
    """
    N_sentences = len(sim_matrix)
    weights = [sum(row) for row in sim_matrix]
    norm_sim_matrix = np.zeros([N_sentences,N_sentences])
    for i in xrange(N_sentences):
        for j in xrange(N_sentences):
            if weights[j]>0:
                norm_sim_matrix[i,j] = sim_matrix[i,j]/weights[j]
    scores = np.ones(N_sentences)  # initialize pageranks
    i=0  # iterations
    while(i<iterations):
        prev_scores = scores
        scores = (1.-alpha)*np.ones(N_sentences) + alpha * norm_sim_matrix.dot(prev_scores)
        if max(abs(scores-prev_scores))<tol:
            return scores
        i += 1
    print 'Max pagerank iterations',iterations,'exceeded'    
    return scores
    
def process_txt_to_html(filename,outfile):
    """
    Processes a text file given by filename
    ranking the sentences via TextRank algorithm
    and writes the results to an html file (outfile)
    with a slider that selectively highlights 
    the sentences in importance
    """
    raw_text = read_textfile(filename)
    sentences = process_text_to_sentences(raw_text)
    stem_sentences = process_sentences_to_stems(sentences, min_word_len=5)
    sim_matrix = generate_similarity_matrix(stem_sentences)
    marked_text = str(raw_text)
    pr = page_rank(sim_matrix,alpha=0.85,iterations=200,tol=1e-8)
    sorted_sents =  sorted(sentences,key=lambda x: dict(zip(sentences,pr))[x],reverse=True)
    for i in range(len(sorted_sents)):
        level = int(np.ceil(100*(i+1)/(1.*len(sorted_sents))))
        ex = sorted_sents[i]
        marked_text = marked_text.replace(ex,'<mark style="background-color:white;"data-highlight="'+str(level)+'">'+ex+'</mark>')

    html_head = '<!DOCTYPE html><html><head><meta charset=utf-8><title>'+filename+'</title></head>'    
    slider = """    
    <label for=fader>Highlight Level</label>
    <input type=range min=0 max=100 value=0 id=fader step=1 onchange="outputUpdate(value)">
    <output for=fader id=highlight>0</output></br>
    """
    js = """
    <script>
    function outputUpdate(level) {
      document.querySelector('#highlight').value = level;
      var sentences = document.querySelectorAll('mark');
      for(i=0; i<sentences.length; i++) {// Cycle through them
        if(parseInt(sentences[i].dataset.highlight,10) <= level) // Change the color based on current level.
          sentences[i].style.backgroundColor  = "yellow"; 
        else
          sentences[i].style.backgroundColor  = "white";
      }
    }
    </script>
    """
    html_body = '<body>'+slider+marked_text+js+'</body></html>'
    with open(outfile,'w') as f:
        f.write(html_head+html_body)
    return dict(zip(sentences,pr))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Highlight some text.')
    parser.add_argument('infile', type=str,
                       help='The input text file')
    args = parser.parse_args()
    filename = args.infile
    try:
        outfile = filename.split('.')[0] +'_hl.html'
        text_ranks = process_txt_to_html(filename,outfile)
        print 'Highlighted html written to:',outfile
    except ValueError as err:
        print(err)
    except IOError as err:
        print(err)
        
