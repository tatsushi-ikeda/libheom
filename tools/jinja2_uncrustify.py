#!/usr/bin/env python3

import re
import argparse
import subprocess as sub
import tempfile
from sys import stdin, stdout, stderr, exit

captured_j2_statement  = []
capture_statement_id   = -1

captured_j2_expression = []
capture_expression_id  = -1

j2_statement_regexp  = r"({%.*?%})"
alt_statement_format = "/*JJ{}{}*/"
alt_statement_regexp = r"/\*JJ([0-9]+) *\*/"
alt_statement_length = 6

j2_expression_regexp  = r"({{.*?}})"
alt_expression_format = "JJ{}{}"
alt_expression_regexp = r"JJ([0-9]+)(_*)"
alt_expression_length = 2

def capture_replace_j2_statement(match):
    global capture_statement_id
    captured_j2_statement.append(match.group())
    capture_statement_id += 1
    j2_length   = len(match.group())
    return alt_statement_format.format(capture_statement_id, ' '*(j2_length - len(str(capture_statement_id)) - alt_statement_length))

def restore_j2_statement(match):
    return captured_j2_statement[int(match.group(1))]

def capture_replace_j2_expression(match):
    global capture_expression_id
    capture_expression_id += 1
    j2_length      = len(match.group())
    filling_length = j2_length - len(str(capture_expression_id)) - alt_expression_length
    captured_j2_expression.append((match.group(), filling_length))
    return alt_expression_format.format(capture_expression_id,
                                        '_'*filling_length)

def restore_j2_expression(match):
    matched_filling_length = len(match.group(2))
    expression, original_filling_length = captured_j2_expression[int(match.group(1))]
    return expression + '_'*(matched_filling_length - original_filling_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="uncrustify")
    parser.add_argument('files', nargs='*')
    parser.add_argument('-f', metavar='FILE')
    parser.add_argument('-c', metavar='CFG')
    parser.add_argument('-L', metavar='SEG')
    parser.add_argument('--command', default='uncrustify')
    languages = ['C', 'CPP', 'D', 'CS', 'JAVA', 'PAWN', 'OC', 'OC+', 'VALA', 'CUDA']
    languages.extend(['J2/'+ l for l in languages])
    parser.add_argument('-l', choices=languages)

    args = parser.parse_args()

    command = [args.command]
    uses_j2 = False

    if args.c:
        command.extend(['-c', args.c])
    if args.L:
        command.extend(['-L', args.L])
    if args.l:
        lang = args.l
        if lang.startswith('J2/'):
            uses_j2 = True
            lang = lang[3:]
        command.extend(['-l', lang])
    else:
        print('You must specify the -l option.', file=stderr)
        exit(1)

    returncode = 0

    def process(input_text, uses_j2, uses_file):
        global capture_statement_id
        global capture_expression_id
        captured_j2_statement.clear()
        captured_j2_expression.clear()
        capture_statement_id  = -1
        capture_expression_id = -1

        if uses_j2:
            input_text = re.sub(j2_expression_regexp, capture_replace_j2_expression, input_text)
            input_text = re.sub(j2_statement_regexp,  capture_replace_j2_statement,  input_text)

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

        if uses_j2:
            output_text = re.sub(alt_statement_regexp,  restore_j2_statement,  output_text)
            output_text = re.sub(alt_expression_regexp, restore_j2_expression, output_text)

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
                stdout.write(process(inp.read(), uses_j2, True))
                stdout.flush
    else:
        stdout.write(process(stdin.read(), uses_j2, False))
        stdout.flush

    exit(returncode)
