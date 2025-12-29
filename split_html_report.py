#!/usr/bin/env python3
"""Split a large HTML report into smaller per-prompt files with accuracy stats."""
import re
import os
import sys
from typing import List, Tuple, Dict


def extract_stats(prompt_content: str) -> Dict:
    """Extract accuracy stats from prompt HTML content.

    Extracted samples: <div class="card extracted"> with "final_q X.XXXX" in header
    Reference trajectories: <div class="card reference"> with "final_q X.XXXX" in header
    """
    stats = {
        'extracted_q': [],
        'reference_q': [],
        'reference_gt': [],  # ground truth rewards
    }

    # Find all extracted cards and their final_q values
    # Pattern: <div class="card extracted">...<div class="hdr">sample X | final_q X.XXXX | tok X</div>
    extracted_pattern = r'<div class="card extracted">.*?<div class="hdr">(.*?)</div>'
    for match in re.finditer(extracted_pattern, prompt_content, re.DOTALL):
        header = match.group(1)
        q_match = re.search(r'final_q ([\d.]+)', header)
        if q_match:
            stats['extracted_q'].append(float(q_match.group(1)))

    # Find all reference cards and their final_q and gt values
    # Pattern: <div class="card reference">...<div class="hdr">ref X | gt X.XXXX | final_q X.XXXX | tok X</div>
    reference_pattern = r'<div class="card reference">.*?<div class="hdr">(.*?)</div>'
    for match in re.finditer(reference_pattern, prompt_content, re.DOTALL):
        header = match.group(1)
        q_match = re.search(r'final_q ([\d.]+)', header)
        gt_match = re.search(r'gt ([\d.]+)', header)
        if q_match:
            stats['reference_q'].append(float(q_match.group(1)))
        if gt_match:
            stats['reference_gt'].append(float(gt_match.group(1)))

    return stats


def compute_accuracy(q_values: List[float], threshold: float = 0.5) -> Tuple[int, int, float]:
    """Compute accuracy: count of q >= threshold."""
    if not q_values:
        return 0, 0, 0.0
    correct = sum(1 for q in q_values if q >= threshold)
    total = len(q_values)
    acc = correct / total if total > 0 else 0.0
    return correct, total, acc


def make_stats_html(stats: Dict, threshold: float = 0.5) -> str:
    """Generate HTML snippet showing accuracy stats."""
    ext_correct, ext_total, ext_acc = compute_accuracy(stats['extracted_q'], threshold)
    ref_correct, ref_total, ref_acc = compute_accuracy(stats['reference_q'], threshold)

    # Also compute ground truth accuracy for reference
    gt_correct, gt_total, gt_acc = compute_accuracy(stats['reference_gt'], threshold)

    html = '<div class="stats">\n'
    html += '<h3>Accuracy Statistics</h3>\n'
    html += '<table style="border-collapse: collapse; margin: 10px 0;">\n'
    html += '<tr><th style="text-align:left; padding: 4px 12px;">Policy</th>'
    html += '<th style="padding: 4px 12px;">Correct</th>'
    html += '<th style="padding: 4px 12px;">Total</th>'
    html += '<th style="padding: 4px 12px;">Accuracy</th></tr>\n'

    if ext_total > 0:
        html += f'<tr><td style="padding: 4px 12px;"><b>Extracted Policy</b></td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{ext_correct}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{ext_total}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;"><b>{ext_acc:.1%}</b></td></tr>\n'

    if ref_total > 0:
        html += f'<tr><td style="padding: 4px 12px;">Reference (Q-pred)</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{ref_correct}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{ref_total}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{ref_acc:.1%}</td></tr>\n'

    if gt_total > 0:
        html += f'<tr><td style="padding: 4px 12px;">Reference (Ground Truth)</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{gt_correct}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{gt_total}</td>'
        html += f'<td style="text-align:center; padding: 4px 12px;">{gt_acc:.1%}</td></tr>\n'

    html += '</table>\n'

    if ext_total > 0 and gt_total > 0:
        improvement = ext_acc - gt_acc
        html += f'<p><b>Improvement over reference: {improvement:+.1%}</b></p>\n'

    html += '</div>\n'
    return html


def split_html(input_file: str, output_dir: str = None, threshold: float = 0.5):
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

    # Aggregate stats
    all_extracted_q = []
    all_reference_q = []
    all_reference_gt = []
    prompt_results = []

    # Write index file
    index_content = header + '<h2>Index</h2>\n'

    for i, prompt_content in enumerate(prompts):
        # Extract prompt idx from content
        idx_match = re.search(r'prompt_idx (\d+)', prompt_content)
        idx = idx_match.group(1) if idx_match else str(i)

        # Extract stats
        stats = extract_stats(prompt_content)
        all_extracted_q.extend(stats['extracted_q'])
        all_reference_q.extend(stats['reference_q'])
        all_reference_gt.extend(stats['reference_gt'])

        ext_correct, ext_total, ext_acc = compute_accuracy(stats['extracted_q'], threshold)
        ref_correct, ref_total, ref_acc = compute_accuracy(stats['reference_q'], threshold)
        gt_correct, gt_total, gt_acc = compute_accuracy(stats['reference_gt'], threshold)

        prompt_results.append({
            'idx': idx,
            'ext_acc': ext_acc,
            'ext_correct': ext_correct,
            'ext_total': ext_total,
            'ref_acc': ref_acc,
            'gt_acc': gt_acc,
            'gt_correct': gt_correct,
            'gt_total': gt_total,
        })

        # Generate stats HTML
        stats_html = make_stats_html(stats, threshold)

        filename = f'prompt_{idx}.html'
        filepath = os.path.join(output_dir, filename)

        # Insert stats after the prompt section header
        modified_content = re.sub(
            r'(<h3>Extracted Policy)',
            stats_html + r'\1',
            prompt_content,
            count=1
        )

        with open(filepath, 'w') as f:
            f.write(header)
            f.write('<hr>')
            f.write(modified_content)
            f.write(footer)

        print(f'Wrote {filepath}')

    # Compute aggregate stats
    agg_ext_correct, agg_ext_total, agg_ext_acc = compute_accuracy(all_extracted_q, threshold)
    agg_ref_correct, agg_ref_total, agg_ref_acc = compute_accuracy(all_reference_q, threshold)
    agg_gt_correct, agg_gt_total, agg_gt_acc = compute_accuracy(all_reference_gt, threshold)

    # Print results
    print('\n' + '='*60)
    print('ACCURACY STATISTICS')
    print('='*60)
    print(f'\nThreshold for "correct": final_q >= {threshold}')
    print('\n--- Per-Prompt Results ---')
    print(f'{"Prompt":<10} {"Extracted":<15} {"Ref (GT)":<15} {"Improvement":<12}')
    print('-'*52)
    for r in prompt_results:
        imp = r['ext_acc'] - r['gt_acc']
        print(f'{r["idx"]:<10} {r["ext_acc"]:>6.1%} ({r["ext_correct"]}/{r["ext_total"]})   '
              f'{r["gt_acc"]:>6.1%} ({r["gt_correct"]}/{r["gt_total"]})   {imp:>+6.1%}')

    print('\n--- Aggregate Results ---')
    print(f'Extracted Policy:     {agg_ext_acc:>6.1%} ({agg_ext_correct}/{agg_ext_total})')
    print(f'Reference (Q-pred):   {agg_ref_acc:>6.1%} ({agg_ref_correct}/{agg_ref_total})')
    print(f'Reference (GT):       {agg_gt_acc:>6.1%} ({agg_gt_correct}/{agg_gt_total})')
    print(f'\nImprovement over reference (GT): {agg_ext_acc - agg_gt_acc:+.1%}')
    print(f'Relative improvement: {agg_ext_acc / agg_gt_acc:.2f}x' if agg_gt_acc > 0 else '')
    print('='*60)

    # Add aggregate stats to index
    index_content += '<div class="stats">\n'
    index_content += '<h3>Aggregate Statistics</h3>\n'
    index_content += '<table style="border-collapse: collapse; margin: 10px 0;">\n'
    index_content += '<tr><th style="text-align:left; padding: 4px 12px;">Policy</th>'
    index_content += '<th style="padding: 4px 12px;">Correct</th>'
    index_content += '<th style="padding: 4px 12px;">Total</th>'
    index_content += '<th style="padding: 4px 12px;">Accuracy</th></tr>\n'
    index_content += f'<tr><td style="padding: 4px 12px;"><b>Extracted Policy</b></td>'
    index_content += f'<td style="text-align:center;">{agg_ext_correct}</td>'
    index_content += f'<td style="text-align:center;">{agg_ext_total}</td>'
    index_content += f'<td style="text-align:center;"><b>{agg_ext_acc:.1%}</b></td></tr>\n'
    index_content += f'<tr><td style="padding: 4px 12px;">Reference (Ground Truth)</td>'
    index_content += f'<td style="text-align:center;">{agg_gt_correct}</td>'
    index_content += f'<td style="text-align:center;">{agg_gt_total}</td>'
    index_content += f'<td style="text-align:center;">{agg_gt_acc:.1%}</td></tr>\n'
    index_content += '</table>\n'
    index_content += f'<p><b>Improvement: {agg_ext_acc - agg_gt_acc:+.1%} '
    index_content += f'({agg_ext_acc / agg_gt_acc:.2f}x)</b></p>\n' if agg_gt_acc > 0 else '</p>\n'
    index_content += '</div>\n'

    # Per-prompt links
    index_content += '<h3>Per-Prompt Results</h3>\n<ul>\n'
    for r in prompt_results:
        filename = f'prompt_{r["idx"]}.html'
        imp = r['ext_acc'] - r['gt_acc']
        index_content += f'<li><a href="{filename}">Prompt {r["idx"]}</a>: '
        index_content += f'Extracted {r["ext_acc"]:.0%} vs Reference {r["gt_acc"]:.0%} '
        index_content += f'({imp:+.0%})</li>\n'

    index_content += '</ul>\n' + footer
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_content)

    print(f'\nSplit into {len(prompts)} files in {output_dir}/')
    print(f'Open {index_path} to browse')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python split_html_report.py <input.html> [output_dir] [threshold]')
        print('  threshold: Q-value threshold for "correct" (default: 0.5)')
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    split_html(input_file, output_dir, threshold)
