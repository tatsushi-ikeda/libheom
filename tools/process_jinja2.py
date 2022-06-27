from jinja2 import Template, Environment, FileSystemLoader
import argparse
import os
import json
import itertools
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('output')
parser.add_argument('params')

args = parser.parse_args()
basename, filename = os.path.split(args.source)

env = Environment(loader=FileSystemLoader(basename))

with open(args.params, 'r') as inp:
    params = json.load(inp)

# define custom filters

def format2(args, f):
    if hasattr(args, "__iter__"):
        params = OrderedDict()
        for i, arg in enumerate(args):
            params[f'n{i}'] = arg
        return f.format(**params)
    else:
        return f.format(args)
env.filters['format2'] = format2

def product(args):
    return list(itertools.product(*args))
env.filters['product'] = product

# define auxiliary functions

def swap(a, b, flag):
    if flag:
        return b, a
    else:
        return a, b
params['swap'] = swap

template = env.get_template(filename)
disp_text = template.render(params)

if args.output:
    with open(args.output, 'w') as out:
        print(disp_text, file=out)
else:
    print(disp_text)
