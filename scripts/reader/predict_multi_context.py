#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json
import datetime
import regex
import pickle
from tqdm import tqdm
from drqa.reader import Predictor
from drqa.retriever import DocDB

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default=None,
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('db_path', type=str, default=None,
                    help='database path')
parser.add_argument('topn', type=int, default=None,
                    help='use topn retrieve results')
parser.add_argument('mode', type=str, default='doc',help='retriever level')
parser.add_argument('--debug', action='store_true',
                    help='Debug with few lines')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='data/preds',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true',
                    help='Only store single top span instead of top N list')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------


questions = []
with open('data/datasets/SQuAD-v1.1-dev.txt') as infp:
    for line in infp:
        questions.append(json.loads(line)['question'])
logger.info("loading questions from data/datasets/SQuAD-v1.1-dev.txt %s" % len(questions))

def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)

db = DocDB(db_path=args.db_path)

def load_doc_level():
    debug = args.debug
    examples = []
    qids = []
    qid2eidxs = {}
    retrieve_scores = []
    with open(args.dataset)  as f:
        for idx_q, line in enumerate(f):
            if debug and idx_q == 10:
                break
            data = eval(line)
            qids.append(data['id'])
            qid2eidxs[data['id']] = []
            doc_cnt = 0
            for is_golden, score, title in list(zip(data['can_recall'], data['can_score'], data['can_title']))[:args.topn]:
                for para in _split_doc(db.get_doc_text(title)):
                    doc_cnt += 1
                    examples.append( (para , questions[idx_q]) )
                    retrieve_scores.append( score )
                    qid2eidxs[data['id']].append( len(examples) - 1 )
    return examples, qids, qid2eidxs, retrieve_scores


def load_para_level():
    debug=args.debug
    examples = []
    qids = []
    qid2eidxs = {}
    retrieve_scores = []
    with open(args.dataset)  as f:
        for idx_q, line in enumerate(f):
            if debug and idx_q == 100:
                break
            data = eval(line)
            qids.append(data['id'])
            qid2eidxs[data['id']] = []
            st = time.time()
            title2text = {xx:yy for xx,yy in db.get_para_text_batch(data['can_title'])}
            for is_golden, score, title in list(zip(data['can_recall'], data['can_score'], data['can_title']))[:args.topn]:
                examples.append( (title2text[title] , questions[idx_q]) )
                retrieve_scores.append( score )
                qid2eidxs[data['id']].append( len(examples) - 1 )
            logger.info("used time %s" % ( time.time() - st ) )
    return examples, qids, qid2eidxs, retrieve_scores


if args.mode == 'doc':
    examples, qids, qid2eidxs, retrieve_scores = load_doc_level()
elif args.mode == 'para':
    examples, qids, qid2eidxs, retrieve_scores = load_para_level()


logger.info("lengths")
logger.info("%s\t%s" % ("examples", len(examples)))
logger.info("%s\t%s" % ("qids", len(qids)))
logger.info("%s\t%s" % ("retrieve_scores", len(retrieve_scores)))
logger.info("%s\t%s" % ("qid2eidxs", len(qid2eidxs)))


logger.info("start predicting")


all_predictions = []
for i in tqdm(range(0, len(examples), args.batch_size)):
    predictions = predictor.predict_batch(
        examples[i:i + args.batch_size], top_n=args.top_n
    )
    for j in range(len(predictions)):
        all_predictions.append( [ (p[0], float(p[1])) for p in predictions[j] ] )


# 根据retrieve分数和reader 分数对答案进行选择
def strategy_one(q_predictions, q_retrieve_score):
    flatten_predictions = sum(q_predictions, [])
    ans,score = "",-1
    for a,s in flatten_predictions:
        if s > score:
            score = s
            ans = a
    return ans


results = {} #official evaluation requires qid -> span
for qid in qids:
    q_predictions  = [all_predictions[x] for x in qid2eidxs[qid]]
    q_retrieve_score = [retrieve_scores[x] for x in qid2eidxs[qid]]
    results[qid] = strategy_one(q_predictions, q_retrieve_score)

# dump data for late analysis
pickle.dump([examples, all_predictions, retrieve_scores,qids,qid2eidxs], open("data/tmp/%s_pred_info.pkl" % args.mode, 'wb'))

model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
time_str = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
outfile = os.path.join(args.out_dir, basename + '-' + model + '-' + args.mode + '-' + time_str + '.preds')
logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    json.dump(results, f)

logger.info('Total time: %.2f' % (time.time() - t0))
