import os


prefix = '/content/data/images/' ######### images folder address + insert 'train_' before images  
dest = ''  
lines = ''
path = os.path.dirname(__file__)
print(path)

with open(path+'/train_set.txt', 'r') as src:
    lines = src.read()
lines = lines.split('\n')
lines = [line.split(' ')[0] for line in lines]

with open(path+'/train_set.txt', 'w') as dest:  
    for line in lines[:-1]:  
        if(line==''):
            continue
        dest.write('%s%s\n' % (prefix, line.rstrip('\n')))

with open(path+'/test_set.txt', 'r') as src:
    lines = src.read()
lines = lines.split('\n')
lines = [line.split(' ')[0] for line in lines]

with open(path+'/test_set.txt', 'w') as dest:  
    for line in lines[:-1]:
        if(line==''):
            continue  
        dest.write('%s%s\n' % (prefix, line.rstrip('\n')))


