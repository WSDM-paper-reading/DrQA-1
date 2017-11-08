#!/usr/bin/env python
# coding: utf-8

import sqlite3
import logging
import json
import os
import tqdm
import re



logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

out_conn = sqlite3.connect('../../data/wikipedia/docs_para.db')
out_c = out_conn.cursor()
out_c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")
conn = sqlite3.connect('../../data/wikipedia/docs.db')
c = conn.cursor()
c.execute("select * from documents")

pairs = []
batch = 20

for i, x in enumerate(c, start=1):
    for ii, xx in enumerate(x[1].split('\n\n')[1:]):
        pid = x[0] + '_' + str(ii)
        text = x[0] + ' ' + xx
        pairs.append((pid,text))
    if i % batch == 0:
        #logger.info("INSERT INTO documents %s" % len(pairs))
        out_c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
        del pairs[:]
        logger.info("processed %s doc" % i)

out_conn.commit()
out_conn.close()
conn.commit()
conn.close()

