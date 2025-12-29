#!/usr/bin/env python3
"""Split a large HTML report into smaller per-prompt files with accuracy stats."""
import re
import os
import sys

def extract_final_q_values(content: str):
    """Extract final_q values for extracted and reference trajectories."""
    extracted_qs = []
    reference_qs = []

    # Find extracted policy section
    extracted_match = re.search(
        r'<h3>Extracted Policy[^<]*</h3>.*?<div class="grid">(.*?)</div>\s*(?=<h3>|<hr>|$)',
        content, re.DOTALL
    )
    if extracted_match:
        cards = re.findall(r'final_q ([\d.]+)', extracted_match.group(1))
        extracted_qs = [float(q) for q in cards]

    # Find reference trajectories section
    reference_match = re.search(
        r'<h3>Reference Trajectories[^<]*</h3>.*?<div class="grid">(.*?)</div>\s*(?=<h3>|<hr>|$)',
        content, re.DOTALL
    )
    if reference_match:
        cards = re.findall(r'final_q ([\d.]+)', reference_match.group(1))
        reference_qs = [float(q) for q in cards]

    return extracted_qs, reference_qs

def calc_accuracy(qs, threshold=0.5):
    """Calculate accuracy (fraction with q >= threshold)."""
    if not qs:
        return 0.0, 0, 0
    correct = sum(1 for q in qs if q >= threshold)
    return correct / len(qs), correct, len(qs)

def make_stats_html(extracted_qs, reference_qs):
    """Generate HTML for accuracy statistics."""
    ext_acc, ext_correct, ext_total = calc_accuracy(extracted_qs)
    ref_acc, ref_correct, ref_total = calc_accuracy(reference_qs)

    improvement = ext_acc - ref_acc
    relative_imp = (ext_acc / ref_acc - 1) * 100 if ref_acc > 0 else 0

    ext_avg_q = sum(extracted_qs) / len(extracted_qs) if extracted_qs else 0
    ref_avg_q = sum(reference_qs) / len(reference_qs) if reference_qs else 0

    return f'''<div class="stats" style="background: #f0f8ff; border: 2px solid #4a90d9; padding: 15px; margin: 15px 0; border-radius: 8px;">
<h3 style="margin-top: 0;">Accuracy Statistics</h3>
<table style="border-collapse: collapse; width: 100%;">
<tr style="background: #e8f4fc;">
    <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ccc;"></th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ccc;">Extracted Policy</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ccc;">Reference</th>
</tr>
<tr>
    <td style="padding: 8px;"><b>Accuracy</b></td>
    <td style="padding: 8px; text-align: center; color: {'green' if ext_acc > ref_acc else 'black'}; font-weight: bold;">{ext_acc:.1%} ({ext_correct}/{ext_total})</td>
    <td style="padding: 8px; text-align: center;">{ref_acc:.1%} ({ref_correct}/{ref_total})</td>
</tr>
<tr style="background: #f9f9f9;">
    <td style="padding: 8px;"><b>Avg Final Q</b></td>
    <td style="padding: 8px; text-align: center;">{ext_avg_q:.3f}</td>
    <td style="padding: 8px; text-align: center;">{ref_avg_q:.3f}</td>
</tr>
<tr>
    <td style="padding: 8px;"><b>Improvement</b></td>
    <td colspan="2" style="padding: 8px; text-align: center; color: {'green' if improvement > 0 else 'red'}; font-weight: bold;">
        {'+' if improvement >= 0 else ''}{improvement:.1%} absolute ({'+' if relative_imp >= 0 else ''}{relative_imp:.1f}% relative)
    </td>
</tr>
</table>
</div>
'''

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

    # Collect all stats for aggregate
    all_extracted_qs = []
    all_reference_qs = []
    prompt_stats = []

    print("\n" + "="*70)
    print("ACCURACY STATISTICS")
    print("="*70)
    print(f"{'Prompt':<12} {'Extracted':<20} {'Reference':<20} {'Improvement':<15}")
    print("-"*70)

    # Write index file
    index_content = header + '<h2>Index</h2>\n'

    for i, prompt_content in enumerate(prompts):
        # Extract prompt idx from content
        idx_match = re.search(r'prompt_idx (\d+)', prompt_content)
        idx = idx_match.group(1) if idx_match else str(i)

        # Extract Q values
        extracted_qs, reference_qs = extract_final_q_values(prompt_content)
        all_extracted_qs.extend(extracted_qs)
        all_reference_qs.extend(reference_qs)

        # Calculate accuracy
        ext_acc, ext_correct, ext_total = calc_accuracy(extracted_qs)
        ref_acc, ref_correct, ref_total = calc_accuracy(reference_qs)
        improvement = ext_acc - ref_acc

        prompt_stats.append({
            'idx': idx,
            'ext_acc': ext_acc,
            'ext_correct': ext_correct,
            'ext_total': ext_total,
            'ref_acc': ref_acc,
            'ref_correct': ref_correct,
            'ref_total': ref_total,
            'improvement': improvement
        })

        # Print stats
        print(f"idx={idx:<8} {ext_acc:>5.1%} ({ext_correct:>2}/{ext_total:<2})       "
              f"{ref_acc:>5.1%} ({ref_correct:>2}/{ref_total:<2})       "
              f"{'+' if improvement >= 0 else ''}{improvement:>+5.1%}")

        # Generate stats HTML
        stats_html = make_stats_html(extracted_qs, reference_qs)

        # Insert stats after prompt heading
        prompt_with_stats = re.sub(
            r'(<h3>Prompt</h3>\s*<pre>.*?</pre>)',
            r'\1\n' + stats_html,
            prompt_content,
            flags=re.DOTALL
        )

        filename = f'prompt_{idx}.html'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(header)
            f.write('<hr>')
            f.write(prompt_with_stats)
            f.write(footer)

    # Print aggregate stats
    print("-"*70)
    agg_ext_acc, agg_ext_correct, agg_ext_total = calc_accuracy(all_extracted_qs)
    agg_ref_acc, agg_ref_correct, agg_ref_total = calc_accuracy(all_reference_qs)
    agg_improvement = agg_ext_acc - agg_ref_acc
    relative_imp = (agg_ext_acc / agg_ref_acc - 1) * 100 if agg_ref_acc > 0 else 0

    print(f"{'TOTAL':<12} {agg_ext_acc:>5.1%} ({agg_ext_correct:>3}/{agg_ext_total:<3})     "
          f"{agg_ref_acc:>5.1%} ({agg_ref_correct:>3}/{agg_ref_total:<3})     "
          f"{'+' if agg_improvement >= 0 else ''}{agg_improvement:>+5.1%}")
    print("="*70)

    print(f"\nAGGREGATE SUMMARY:")
    print(f"  Extracted Policy Accuracy: {agg_ext_acc:.1%} ({agg_ext_correct}/{agg_ext_total})")
    print(f"  Reference Policy Accuracy: {agg_ref_acc:.1%} ({agg_ref_correct}/{agg_ref_total})")
    print(f"  Absolute Improvement:      {'+' if agg_improvement >= 0 else ''}{agg_improvement:.1%}")
    print(f"  Relative Improvement:      {'+' if relative_imp >= 0 else ''}{relative_imp:.1f}%")
    print(f"  Avg Extracted Final Q:     {sum(all_extracted_qs)/len(all_extracted_qs):.3f}")
    print(f"  Avg Reference Final Q:     {sum(all_reference_qs)/len(all_reference_qs):.3f}")

    # Build index with stats table
    index_content += '''
<div class="stats" style="background: #f0f8ff; border: 2px solid #4a90d9; padding: 15px; margin: 15px 0; border-radius: 8px;">
<h3 style="margin-top: 0;">Aggregate Statistics</h3>
<table style="border-collapse: collapse; width: 100%;">
<tr style="background: #e8f4fc;">
    <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ccc;"></th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ccc;">Extracted Policy</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ccc;">Reference</th>
</tr>
<tr>
    <td style="padding: 8px;"><b>Overall Accuracy</b></td>
    <td style="padding: 8px; text-align: center; color: green; font-weight: bold;">''' + f'{agg_ext_acc:.1%} ({agg_ext_correct}/{agg_ext_total})' + '''</td>
    <td style="padding: 8px; text-align: center;">''' + f'{agg_ref_acc:.1%} ({agg_ref_correct}/{agg_ref_total})' + '''</td>
</tr>
<tr style="background: #f9f9f9;">
    <td style="padding: 8px;"><b>Avg Final Q</b></td>
    <td style="padding: 8px; text-align: center;">''' + f'{sum(all_extracted_qs)/len(all_extracted_qs):.3f}' + '''</td>
    <td style="padding: 8px; text-align: center;">''' + f'{sum(all_reference_qs)/len(all_reference_qs):.3f}' + '''</td>
</tr>
<tr>
    <td style="padding: 8px;"><b>Improvement</b></td>
    <td colspan="2" style="padding: 8px; text-align: center; color: green; font-weight: bold;">
        ''' + f"{'+' if agg_improvement >= 0 else ''}{agg_improvement:.1%} absolute ({'+' if relative_imp >= 0 else ''}{relative_imp:.1f}% relative)" + '''
    </td>
</tr>
</table>
</div>

<h3>Per-Prompt Results</h3>
<table style="border-collapse: collapse; width: 100%;">
<tr style="background: #e8f4fc;">
    <th style="padding: 8px; border: 1px solid #ccc;">Prompt</th>
    <th style="padding: 8px; border: 1px solid #ccc;">Extracted Acc</th>
    <th style="padding: 8px; border: 1px solid #ccc;">Reference Acc</th>
    <th style="padding: 8px; border: 1px solid #ccc;">Improvement</th>
</tr>
'''
    for ps in prompt_stats:
        color = 'green' if ps['improvement'] > 0 else ('red' if ps['improvement'] < 0 else 'black')
        index_content += f'''<tr>
    <td style="padding: 8px; border: 1px solid #ccc;"><a href="prompt_{ps['idx']}.html">idx={ps['idx']}</a></td>
    <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">{ps['ext_acc']:.1%} ({ps['ext_correct']}/{ps['ext_total']})</td>
    <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">{ps['ref_acc']:.1%} ({ps['ref_correct']}/{ps['ref_total']})</td>
    <td style="padding: 8px; border: 1px solid #ccc; text-align: center; color: {color}; font-weight: bold;">{'+' if ps['improvement'] >= 0 else ''}{ps['improvement']:.1%}</td>
</tr>
'''
    index_content += '</table>\n' + footer

    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_content)

    print(f"\nWrote {len(prompts)} files to {output_dir}/")
    print(f"Open {index_path} to browse")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python split_html_report.py <input.html> [output_dir]')
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    split_html(input_file, output_dir)
