#!/usr/bin/env python3
"""Split a large HTML report into smaller per-prompt files."""
import re
import os
import sys

def split_html(input_file: str, output_dir: str = None):
    if output_dir is None:
        output_dir = input_file.replace('.html', '_split')

    with open(input_file, 'r') as f:
        content = f.read()

    # Extract header (everything before first <hr>)
    header_match = re.search(r'^(.*?)<hr>', content, re.DOTALL)
    header = header_match.group(1) if header_match else ''

    # Footer
    footer = '</body>\n</html>\n'

    # Split by prompts (each starts with <hr>)
    parts = re.split(r'<hr>', content)
    prompts = [p for p in parts[1:] if p.strip() and '<h2>Prompt' in p]

    os.makedirs(output_dir, exist_ok=True)

    # Write index file
    index_content = header + '<h2>Index</h2>\n<ul>\n'

    for i, prompt_content in enumerate(prompts):
        # Extract prompt idx from content
        idx_match = re.search(r'prompt_idx (\d+)', prompt_content)
        idx = idx_match.group(1) if idx_match else str(i)

        filename = f'prompt_{idx}.html'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(header)
            f.write('<hr>')
            f.write(prompt_content)
            f.write(footer)

        # Extract some stats for index
        final_q_matches = re.findall(r'final_q ([\d.]+)', prompt_content)
        if final_q_matches:
            avg_q = sum(float(q) for q in final_q_matches) / len(final_q_matches)
            index_content += f'<li><a href="{filename}">Prompt {idx}</a> (avg final_q: {avg_q:.3f})</li>\n'
        else:
            index_content += f'<li><a href="{filename}">Prompt {idx}</a></li>\n'

        print(f'Wrote {filepath}')

    index_content += '</ul>\n' + footer
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_content)

    print(f'\nSplit into {len(prompts)} files in {output_dir}/')
    print(f'Open {index_path} to browse')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python split_html_report.py <input.html> [output_dir]')
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    split_html(input_file, output_dir)
