# -*- encoding:utf-8 -*-
"""
处理多字target
"""
from util import read_lines
r_path = './train/restaurants.txt'
w_path = './train/res_norm.txt'
lines = read_lines(r_path)
docs = []
w_fp = open(w_path, 'w')
for line in lines:
    s_list = line.split('|')
    sentence_id = s_list[0]
    target = s_list[1]
    pol = s_list[2]
    sentence = s_list[3]
    if ' ' in target:
        new_target = '-'.join(target.split())
        # print(new_target)
        sentence = new_target.join(sentence.split(target))
        # print(sentence)
        target = new_target
    string = sentence_id + '|' + target + '|' + pol + '|' + sentence + '\n'
    w_fp.write(string)
w_fp.close()
