path = './SemEval2014/Restaurants_Train.xml'
w_path = './train/restaurants.txt'
from collections import defaultdict
lines = []
with open(path, 'r') as fp:
    lines = fp.readlines()
count = 0
f_count = 0
pos_count = 0
neg_count = 0
neu_count = 0
f_pos = 0
f_neg = 0
f_neu = 0
temp = []
cate_list = defaultdict(int)
pol_list = ['positive', 'negative', 'neutral']
target_list = defaultdict(int)
target_pol = {}
w_fp = open(w_path, 'w', encoding='utf-8')
for line in lines:
    line = line.strip()
    if line:
        if 'id=\"' in line:
            sid = line.split('\"')[1]
            # sid = sid[1:-2]
        if line == '</sentence>':
            if len(target_pol) and len(temp) == 1:
                # print(temp)
                count += 1
                for target in target_pol:
                    pol = target_pol[target]
                    target_list[target] += 1
                    if pol not in pol_list:
                        continue
                    string = sid + '|' + target + '|' + target_pol[target] + '|' + temp[0] + '\n'
                    w_fp.write(string)
                    f_count += 1
                    if pol == 'positive':
                        pos_count += 1
                    elif pol == 'negative':
                        neg_count += 1
                    elif pol == 'neutral':
                        neu_count += 1
                    else:
                        print(sid)
                        print(pol)
                        print('error')
                        exit(0)
            temp = []
            target_pol = {}
            continue
        if '<text>' in line:
            string = line[6:-7]
            string = string.strip()
            temp.append(string)
        elif 'from' in line:
            s_list = line.split('\"')
            target = s_list[1]
            pol = s_list[3]
            target_pol[target] = pol
        if 'category=' in line:
            words_list = line.split('\"')
            cate = words_list[1]
            cate_list[cate] += 1
w_fp.close()
print(count)
print(f_count)
print(pos_count, neg_count, neu_count)
target_list = sorted(target_list.items(), key = lambda a:a[1], reverse=True)
for item in target_list:
    if item[1] > 4:
        print(item[0], item[1])
cate_list = sorted(cate_list.items(), key = lambda a:a[1], reverse=True)
print('-----------------category------------------------')
for item in cate_list:
    print(item[0], item[1])
# print(target_list)
