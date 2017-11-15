#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
import sqlite3
from multiprocessing import Pool as ProcessPool
import tqdm
from multiprocessing.util import Finalize
from itertools import repeat
from tqdm import tqdm
from drqa.retriever import DocDB
from drqa import tokenizers





PROCESS_TOK = None
PROCESS_DB = None

def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)

def ids(toks,word_dict):
    return [str(word_dict.get(tok,0)) for tok in toks]


processes = ProcessPool(
        30,
        initializer=init,
        initargs=(tokenizers.get_class('spacy'), {}, DocDB, {'db_path':"/users/sulixin/relate_work/DrQA/DrQA/data/wikipedia/docs_para.db"})
    )

def load_qa(fi):
    infp_qa = open(fi)
    questions = []
    answers = []
    for l in infp_qa:
        if not l.strip():
            continue
        data = json.loads(l)
        questions.append(data['question'])
        answers.append(data['answer'])
    return questions, answers

def load_retrieval(fi, topn=50):
    titles = []
    recalls = []
    with open(fi) as infp:
        for l in infp:
            if not l.strip():
                continue
            data = eval(l)
            titles.append(data['can_title'][:topn])
            recalls.append(data['can_recall'][:topn])
    return titles, recalls


def load_passages(titles):
    #global processes
    flat_titles = list({t for ts in titles for t in ts})
    title2idx = {t:i for i,t in enumerate(flat_titles)}
    print("all para %s " %len( flat_titles))
    doc_texts = []
    db = sqlite3.connect("/users/sulixin/relate_work/DrQA/DrQA/data/wikipedia/docs_para.db")
    cur = db.cursor()
    step = 100
    for i in tqdm(range(0, len(flat_titles), step)):
        cur.execute('SELECT * FROM document WHERE id in ({0})'.format(', '.join('?' for _ in flat_titles[i:i+step])), flat_titles[i:i+step])
        results = dict(cur.fetch_all())
        if len(results) != len(flat_titles[i:i+step]):
            print("query database miss results")
        doc_texts.extend([results[x] for x in flat_titles[i:i+step]])
    assert len(flat_titles) == len(doc_texts)
    #doc_texts = processes.map(fetch_text, flat_titles)
    return title2idx, doc_texts


def saving_corpus(corpus, word_dict, fo):
    fo = open(fo, 'w')
    for text in corpus:
        pass

if __name__ == "__main__":
    print("load qa pairs")
    questions, answers = load_qa("dev_qa.json")
    print("load retrieval results")
    titles, recalls = load_retrieval("dev_retrival_para_bigram_k0.4_b0.0.json")
    print("parallel load texts")
    title2idx, doc_texts = load_passages(titles)
    print("tokenize texts")
    q_tokens = processes.map_async(tokenize_text, questions)
    p_tokens = processes.map_async(tokenize_text, doc_texts)
    q_tokens = q_tokens.get()
    p_tokens = p_tokens.get()
    #print(q_tokens[0].words())
    #print(p_tokens[0].words())
    q_tokens = [x.words() for x in q_tokens]
    p_tokens = [x.words() for x in p_tokens]
    del doc_texts
    print("generate word dict and saving")
    all_tokens = q_tokens + p_tokens
    tok2cnt = Counter([xx for x in all_tokens for xx in x])
    fo = open('word_freq.txt','w')
    for k,v in tok2cnt.items():
        fo.write("%s\t%s\n" % (k,v))
    print("total words %s" % len(tok2cnt))
    # generate word -> id  id=0 for unknown
    word_dict = {x[0]:i+1 for i,x in enumerate(tok2cnt.most_common(step0))}
    print("saving words %s" % len(word_dict))
    fo = open('word_dict.txt', 'w')
    for k,v in word_dict.items():
        fo.write("%s\t%s\n" % (k, v))
    # transform token to ids
    print("parallel transform token to ids")
    with ProcessPool(processes=30) as pool:
        q_ids = pool.starmap(ids, zip(q_tokens, repeat(word_dict)))
    with ProcessPool(processes=30) as pool:
        p_ids = pool.starmap(ids, zip(p_tokens, repeat(word_dict)))
    del q_tokens
    del p_tokens
    print("saving corpus")
    assert len(q_ids) == len(questions)
    with open("qid_query.txt", 'w') as ofp:
        for i,q in enumerate(q_ids):
            ofp.write("%s %s %s\n" % ( i, len(q), ' '.join(q) ))
    with open('docid_doc.txt', 'w') as ofp:
        for i,doc in enumerate(p_ids):
            ofp.write("%s %s %s\n" % (i, len(doc), ' '.join(doc)))
    # saving relation
    print("saving relation file")
    fo = open("relation.train", 'w')
    for i,data in enumerate(zip(titles, recalls)):
        # pairwise
        data = zip(*data)
        data = sorted(data,key=lambda x:x[1],reverse=True)
        for t,r in data:
            fo.write("%s %s %s\n" % (r, i, title2idx[t]))






