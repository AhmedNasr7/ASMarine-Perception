import os
import random
import time

path = ('/').join(os.path.abspath(__file__).split('/')[:-1])+'/'
lines = []
with open(path+'source.txt','r') as train_set :
    lines = train_set.read().split('\n')

count = {'g1':0 , 'g2':0 ,'g3':0 , 'phone':0 , 'bottle':0 , 'badge': 0 , 'paper':0 ,'tommygun':0 , \
    'cash': 0} 
new_lines = []

gate_items = ['g1' , 'g2' ,'g3']

random.shuffle(lines)
for line in lines:
    for item , num in count.items():
        if num > 1000 or (item in gate_items and num >333):
            continue
        if item in line:
            count[item]+=1
            new_lines.append(line)
    if not line.split('/')[-1].split('.')[0].islower():
        new_lines.append(line) 
        

random.shuffle(new_lines)
chunk  = ('\n').join(new_lines)
with open(path+'train_set.txt','w+') as f:
    t = time.time()
    f.write(chunk)
    print('writing whole block at a time takes %f s'%(time.time()-t))
