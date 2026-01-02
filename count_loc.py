#!/usr/bin/env python3
"""Count lines of code in tinyvllm directory."""

import os
from pathlib import Path

def count_lines(file_path):
    """Count non-blank, non-comment lines in a Python file."""
    total = 0
    code = 0
    comments = 0
    blank = 0
    in_docstring = False
    docstring_char = None
    
    with open(file_path, 'r') as f:
        for line in f:
            total += 1
            stripped = line.strip()
            
            # Track docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                        comments += 1  # Single-line docstring
                    else:
                        in_docstring = True
                        comments += 1
                    continue
            else:
                comments += 1
                if docstring_char in stripped:
                    in_docstring = False
                continue
            
            if not stripped:
                blank += 1
            elif stripped.startswith('#'):
                comments += 1
            else:
                code += 1
    
    return total, code, comments, blank

def main():
    tinyvllm_dir = Path('tinyvllm')
    
    results = []
    total_all = 0
    code_all = 0
    
    for py_file in sorted(tinyvllm_dir.rglob('*.py')):
        if '__pycache__' in str(py_file):
            continue
        total, code, comments, blank = count_lines(py_file)
        rel_path = py_file.relative_to(tinyvllm_dir)
        results.append((str(rel_path), total, code, comments, blank))
        total_all += total
        code_all += code
    
    # Print results
    print(f"{'File':<45} {'Total':>7} {'Code':>7} {'Docs':>7} {'Blank':>7}")
    print("-" * 75)
    
    for path, total, code, comments, blank in results:
        print(f"{path:<45} {total:>7} {code:>7} {comments:>7} {blank:>7}")
    
    print("-" * 75)
    print(f"{'TOTAL':<45} {total_all:>7} {code_all:>7}")
    print()
    print(f"Lines of Code (excluding docs/comments/blanks): {code_all}")
    print(f"Target: 1000 LOC")
    print(f"Remaining: {1000 - code_all} LOC")

if __name__ == '__main__':
    main()
