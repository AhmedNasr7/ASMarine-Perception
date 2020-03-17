import os

prefix = 'blablabla'  
suffix = 'gugugugugu'  
dest = ''  
lines = ''
with open(os.getcwd()+'/file.txt', 'r') as src:
    lines = src.read()
lines = lines.split('\n')
lines = lines[:-1]

with open(os.getcwd()+'/file.txt', 'w') as dest:  
    for line in lines:  
        dest.write('%s %s\n' % (prefix, line.rstrip('\n')))
