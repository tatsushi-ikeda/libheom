#!/usr/bin/env python3

import re
import argparse
import subprocess as sub
import tempfile
from sys import stdin, stdout, stderr, exit

captured_cuda_expression = []
capture_expression_id    = -1

cuda_expression_regexp = r"<<<.*?>>>"
alt_expression_format  = "<CUDA{}{}>"
alt_expression_regexp  = r"<CUDA([0-9]+)(_*)>"
alt_expression_length  = 6

def capture_replace_cuda_expression(match):
    global capture_expression_id
    capture_expression_id += 1
    cuda_length    = len(match.group())
    filling_length = cuda_length - len(str(capture_expression_id)) - alt_expression_length
    captured_cuda_expression.append((match.group(), filling_length))
    return alt_expression_format.format(capture_expression_id,
                                        '_'*filling_length)

def restore_cuda_expression(match):
    matched_filling_length = len(match.group(2))
    expression, original_filling_length = captured_cuda_expression[int(match.group(1))]
    return expression + '_'*(matched_filling_length - original_filling_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="uncrustify")
    parser.add_argument('files', nargs='*')
    parser.add_argument('-f', metavar='FILE')
    parser.add_argument('-c', metavar='CFG')
    parser.add_argument('-L', metavar='SEG')
    parser.add_argument('--command', default='uncrustify')
    languages = ['C', 'CPP', 'D', 'CS', 'JAVA', 'PAWN', 'OC', 'OC+', 'VALA', 'CUDA']
    parser.add_argument('-l', choices=languages)
    args = parser.parse_args()

    command = [args.command]
    is_cuda = False

    if args.c:
        command.extend(['-c', args.c])
    if args.L:
        command.extend(['-L', args.L])
    if args.l:
        lang = args.l
        if lang == 'CUDA':
            is_cuda = True
            lang = 'CPP'
        command.extend(['-l', lang])
    else:
        print('You must specify the -l option.', file=stderr)
        exit(1)

    returncode = 0

    def process(input_text, uses_file):
        global capture_expression_id
        captured_cuda_expression.clear()
        capture_expression_id = -1

        if is_cuda:
            input_text = re.sub(cuda_expression_regexp, capture_replace_cuda_expression, input_text)

        if uses_file:
            with tempfile.NamedTemporaryFile('w') as input:
                input.write(input_text)
                input.flush()
                uncrustify_result = sub.run(command + ['-f', input.name],
                                            stdout=sub.PIPE,
                                            stderr=sub.PIPE,
                                            input=input_text,
                                            encoding='utf-8')
        else:
            uncrustify_result = sub.run(command,
                                        stdout=sub.PIPE,
                                        stderr=sub.PIPE,
                                        input=input_text,
                                        encoding='utf-8')

        output_text = uncrustify_result.stdout
        if uncrustify_result.returncode != 1:
            returncode = uncrustify_result.returncode

        if is_cuda:
            output_text = re.sub(alt_expression_regexp, restore_cuda_expression, output_text)

        if uncrustify_result.stderr and len(uncrustify_result.stderr) > 0:
            stderr.write(uncrustify_result.stderr)
            stderr.flush()
        return output_text

    files = []
    if args.f:
        files.append(args.f)
    if args.files:
        files.extend(args.files)

    if len(files) > 0:
        for file in files:
            with open(file, 'r') as inp:
                stdout.write(process(inp.read(), True))
                stdout.flush
    else:
        stdout.write(process(stdin.read(), False))
        stdout.flush

    exit(returncode)
