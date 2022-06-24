from jinja2 import Template, Environment, FileSystemLoader
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('output')
parser.add_argument('params')

args = parser.parse_args()
basename, filename = os.path.split(args.source)

env = Environment(loader=FileSystemLoader(basename))
template = env.get_template(filename)

def swap(a, b, flag):
    if flag:
        return b, a
    else:
        return a, b

with open(args.params, 'r') as inp:
    params = json.load(inp)

params['swap'] = swap

disp_text = template.render(params)
if args.output:
    with open(args.output, 'w') as out:
        print(disp_text, file=out)
else:
    print(disp_text)
