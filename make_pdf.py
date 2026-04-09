"""Generate PDF report from experiment results using fpdf2."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF


# ---------------------------------------------------------------------------
# Helper: generate the sampling-strategy comparison figure (Fig 5)
# ---------------------------------------------------------------------------
def generate_sampling_comparison_figure(expC, output_dir='results'):
    """
    Bar chart comparing equivariance error across sampling strategies
    (GL_1x, GL_2x, GL_3x, Leb_min, Leb_2x, Uniform_50) for each activation
    at l_max = 6.  Bars are grouped by activation; one colour per method.
    """
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    lmax = '6'
    data = expC[lmax]
    act_names = ['Softplus_1', 'SiLU', 'GELU', 'Softplus_10',
                 'ReLU', 'Softplus_100', 'tanh', 'abs']
    act_names = [a for a in act_names if a in data]

    # Sampling configs to show, with display labels including point count
    cfg_keys = ['GL_1x', 'GL_2x', 'GL_3x', 'Leb_min_d13', 'Leb_2x_d25', 'Uniform_50']
    cfg_labels = ['GL 1x', 'GL 2x', 'GL 3x', 'Leb min', 'Leb 2x', 'Uniform']
    cfg_colors = ['#1f77b4', '#2ca02c', '#17becf', '#ff7f0e', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(act_names))
    n_cfg = len(cfg_keys)
    width = 0.13

    for i, (cfg, label, color) in enumerate(zip(cfg_keys, cfg_labels, cfg_colors)):
        vals, errs = [], []
        for act in act_names:
            if cfg in data[act]:
                vals.append(data[act][cfg]['mean_equiv_error'])
                errs.append(data[act][cfg]['std_equiv_error'])
            else:
                vals.append(0)
                errs.append(0)
        n_pts = data[act_names[0]].get(cfg, {}).get('n_points', '?')
        ax.bar(x + i * width, vals, width, yerr=errs, label=f'{label} (N={n_pts})',
               color=color, alpha=0.85, capsize=2)

    ax.set_xticks(x + width * (n_cfg - 1) / 2)
    ax.set_xticklabels(act_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Equivariance Error', fontsize=10)
    ax.set_yscale('log')
    ax.set_title('Equivariance Error by Sampling Strategy ($\\ell_{\\max}=6$)', fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = f'{output_dir}/figures/fig5_sampling_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# PDF Report class
# ---------------------------------------------------------------------------
class Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'S2 Activation: Spectral Leakage and Equivariance in SO(3)-Equivariant Networks', 0, 1, 'C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.ln(5)
        self.cell(0, 10, title, 0, 1)
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.ln(3)
        self.cell(0, 7, title, 0, 1)
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, 1, 0, 'C', True)
        self.ln()
        self.set_font('Helvetica', '', 8)
        self.set_text_color(0, 0, 0)
        for row_idx, row in enumerate(rows):
            fill = row_idx % 2 == 0
            if fill:
                self.set_fill_color(240, 245, 250)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5, str(val), 1, 0, 'C', fill)
            self.ln()
        self.ln(2)

    def add_figure(self, path, caption='', w=180):
        if os.path.exists(path):
            if self.get_y() + 80 > 270:
                self.add_page()
            self.image(path, x=15, w=w)
            if caption:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(80, 80, 80)
                self.multi_cell(0, 4, caption)
            self.ln(3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load all experiment metrics
    with open('results/metrics/expA_spectral.json') as f:
        expA = json.load(f)
    with open('results/metrics/expB_coefficient_error.json') as f:
        expB = json.load(f)
    with open('results/metrics/expC_equivariance.json') as f:
        expC = json.load(f)
    with open('results/metrics/expD_task.json') as f:
        expD = json.load(f)
    with open('results/metrics/expE_expressibility.json') as f:
        expE = json.load(f)

    # Generate the new sampling-comparison figure
    fig5_path = generate_sampling_comparison_figure(expC)

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ==================================================================
    # TITLE PAGE
    # ==================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, 'S2 Activation:', 0, 1, 'C')
    pdf.cell(0, 12, 'Spectral Leakage and Equivariance', 0, 1, 'C')
    pdf.cell(0, 12, 'in SO(3)-Equivariant Networks', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, 'A Numerical Study of Nonlinearity and Quadrature Effects', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, 'Experiments run on NERSC Perlmutter', 0, 1, 'C')

    # ==================================================================
    # 1. INTRODUCTION
    # ==================================================================
    pdf.add_page()
    pdf.section_title('1. Introduction')

    pdf.body_text(
        'S2 Activation is the standard pointwise nonlinearity in SO(3)-equivariant neural '
        'networks, including SCN, eSCN, EquiformerV2, and related architectures. '
        'Given a set of spherical harmonic (SH) coefficients, it applies a scalar '
        'nonlinearity on the sphere and projects the result back to the SH basis. '
        'While the operation is exact in the limit of infinite bandwidth and infinite '
        'quadrature, practical implementations truncate the SH expansion at a maximum '
        'degree l_max and approximate the sphere integral with a finite quadrature rule. '
        'Both approximations introduce error and can break SO(3) equivariance.\n\n'
        'This report investigates two orthogonal axes that control these errors:\n'
        '  (i)  the choice of activation function (Section 2), and\n'
        '  (ii) the choice of quadrature / sampling strategy (Section 3).\n'
        'We find that these two axes affect the system through fundamentally different '
        'mechanisms -- truncation error vs. aliasing error -- leading to a surprising '
        'dichotomy in their effects on coefficient accuracy vs. equivariance.'
    )

    pdf.sub_title('1.1 The S2 Activation Pipeline')
    pdf.body_text(
        'Let f : S^2 -> R be a band-limited function represented by SH coefficients '
        '{c_lm} for l = 0, ..., l_max and m = -l, ..., l. The S2 Activation applies '
        'a pointwise nonlinearity sigma in three steps:\n\n'
        'Step 1 (Forward transform): Evaluate f on a set of N quadrature points '
        '{x_i} on S^2:\n'
        '    f(x_i) = sum_{l=0}^{l_max} sum_{m=-l}^{l} c_lm Y_lm(x_i)\n'
        'where Y_lm are real spherical harmonics.\n\n'
        'Step 2 (Pointwise nonlinearity): Compute g(x_i) = sigma(f(x_i)) at each point.\n\n'
        'Step 3 (Inverse transform): Project back to SH coefficients via quadrature:\n'
        '    c\'_lm = sum_{i=1}^{N} w_i g(x_i) Y_lm(x_i)\n'
        'where {w_i} are the quadrature weights (summing to 4*pi).'
    )

    pdf.sub_title('1.2 Definitions')
    pdf.body_text(
        'Angular power spectrum.  For a function with SH coefficients {c_lm}, '
        'the power at degree l is:\n'
        '    P(l) = sum_{m=-l}^{l} |c_lm|^2\n'
        'This measures the total energy at angular frequency l.\n\n'
        'Spectral leakage ratio.  After applying sigma to a band-limited input '
        '(l <= l_max), the output g generally has energy at all degrees. We define '
        'the leakage ratio as the fraction of output energy above the input bandwidth:\n'
        '    R = sum_{l > l_max} P_g(l) / sum_{l >= 0} P_g(l)\n'
        'where P_g(l) is the power spectrum of g, computed using a high-resolution '
        'reference quadrature (l_max_ref = 25). R = 0 means no leakage; R = 1 means '
        'all energy leaked.\n\n'
        'Coefficient error.  Let c\'_out be the output SH coefficients from S2 Activation '
        'with a given quadrature, and c\'_gt the ground-truth coefficients obtained with '
        'a high-resolution reference quadrature (GL at l_max_ref = 25, truncated to l_max). '
        'The relative coefficient error is:\n'
        '    E_coeff = ||c\'_out - c\'_gt|| / ||c\'_gt||\n\n'
        'Equivariance error.  For a rotation R in SO(3) with Wigner-D matrix D(R), '
        'exact equivariance requires S2Act(D(R) c) = D(R) S2Act(c). We measure the '
        'relative violation:\n'
        '    E_equiv = ||S2Act(D(R) c) - D(R) S2Act(c)|| / ||S2Act(D(R) c)||\n'
        'averaged over 20 random rotations and 10 random inputs.'
    )

    pdf.sub_title('1.3 Experimental Setup')
    pdf.body_text(
        'All experiments use random band-limited inputs: each SH coefficient c_lm is '
        'drawn i.i.d. from N(0,1), giving (l_max + 1)^2 total coefficients. Results '
        'are averaged over 10 random inputs (seeds 0-9). We test l_max in {3, 6, 10} '
        'and focus on l_max = 6 in the main text.\n\n'
        'Activation functions tested:\n'
        '  - C^inf smooth: Softplus(beta=1), SiLU, GELU, tanh\n'
        '  - C^0 (non-differentiable at origin): ReLU, abs, LeakyReLU\n'
        '  - Parametric family: Softplus(beta) with beta in {1, 3, 10, 30, 100}\n'
        '    (Softplus(beta, x) = (1/beta) ln(1 + exp(beta x)); approaches ReLU as '
        'beta -> inf)\n'
        '  - Special: x^2 (exactly band-limited to 2 l_max), sin\n\n'
        'Quadrature methods tested:\n'
        '  - Gauss-Legendre (GL): n_theta GL nodes in cos(theta) x n_phi uniform '
        'in phi.\n'
        '    GL_kx denotes k-times oversampling (n_theta = k (l_max+1)).\n'
        '  - Lebedev: Symmetric quadrature exact for polynomials up to a given '
        'algebraic degree.\n'
        '  - Uniform (equiangular): Equally spaced grid in (theta, phi).'
    )

    # ==================================================================
    # 2. EQUIVARIANCE ERROR VS. ACTIVATION FUNCTION
    # ==================================================================
    pdf.add_page()
    pdf.section_title('2. Equivariance Error vs. Activation Function')

    pdf.body_text(
        'We first investigate how the choice of nonlinearity affects equivariance, '
        'holding the quadrature fixed. The key finding is that smoother activation '
        'functions produce less spectral leakage, and that leakage is strongly '
        'correlated with equivariance error.'
    )

    pdf.sub_title('2.1 Spectral Leakage Across Activations')
    pdf.body_text(
        'Figure 1 shows the spectral leakage ratio R for each activation function '
        'at l_max = 3, 6, and 10. The leakage ratio measures the fraction of output '
        'energy that falls above the input bandwidth l_max (see Section 1.2). '
        'This energy is irrecoverably lost upon truncation back to l_max coefficients.\n\n'
        'Activations are sorted from lowest to highest leakage. The pattern is '
        'consistent across all l_max values:\n'
        '  - C^inf activations (Softplus_1, SiLU, GELU) have the lowest leakage '
        '(R ~ 0.01-0.08), because their smoothness produces rapidly decaying high-l '
        'Fourier content.\n'
        '  - C^0 activations (ReLU, abs) have high leakage (R ~ 0.09-0.16), because '
        'the cusp at the origin generates power-law spectral tails.\n'
        '  - The Softplus(beta) family interpolates continuously between these '
        'extremes, confirming that the effect is controlled by activation smoothness.\n'
        '  - The polynomial x^2 has zero leakage beyond 2 l_max by construction '
        '(its output is exactly band-limited).\n'
        '  - tanh shows moderate leakage despite being C^inf. This is because tanh '
        'saturates to +/-1, compressing the output range and concentrating energy '
        'differently from non-saturating smooth activations.'
    )

    # Fig 1: Leakage ratio bar chart (= old A4)
    pdf.add_figure(
        'results/figures/expA_leakage_ratios.png',
        'Figure 1: Spectral leakage ratio R = sum_{l>l_max} P(l) / sum_l P(l) '
        'for each activation function, measured at l_max = 3, 6, 10. '
        'Input: 10 random SH coefficient vectors (c_lm ~ N(0,1)). '
        'Output spectrum computed via GL quadrature at l_max_ref = 25. '
        'Red bars: C^0 activations. Green: Softplus family. Blue: C^inf smooth. '
        'Error bars: standard deviation across 10 inputs.'
    )

    pdf.sub_title('2.2 Leakage Predicts Equivariance Error')
    pdf.body_text(
        'Figure 2 plots the equivariance error E_equiv (measured in Experiment C) '
        'against the spectral leakage ratio R (measured in Experiment A) for each '
        'activation at l_max = 6, using GL_1x quadrature.\n\n'
        'The strong positive correlation confirms the mechanistic link: activations '
        'that leak more energy beyond l_max suffer larger equivariance violations, '
        'because the leaked energy aliases back into the l <= l_max coefficients '
        'in a rotation-dependent manner. Since different rotations move the quadrature '
        'points to different positions on the sphere, the aliasing pattern changes '
        'with rotation, breaking equivariance.'
    )

    # Fig 2: Equivariance error vs leakage correlation (= old C3)
    pdf.add_figure(
        'results/figures/expC_equiv_vs_leakage.png',
        'Figure 2: Equivariance error E_equiv vs. spectral leakage ratio R for '
        'each activation at l_max = 3, 6, 10 using GL_1x quadrature. '
        'E_equiv = ||S2Act(Dc) - D S2Act(c)|| / ||S2Act(Dc)||, averaged over 20 '
        'random SO(3) rotations and 10 random inputs. Each point is one activation. '
        'The correlation shows that spectral leakage is a strong predictor of '
        'equivariance violation.',
        w=140
    )

    # Combined table
    pdf.sub_title('Table 1: Leakage and Equivariance Error (l_max = 6, GL_1x)')
    headers = ['Activation', 'Leakage R', 'Equiv Error', 'Smoothness']
    rows = []
    act_order = ['Softplus_1', 'SiLU', 'GELU', 'Softplus_10', 'ReLU',
                 'Softplus_100', 'tanh', 'abs']
    smooth_label = {
        'Softplus_1': 'C^inf', 'SiLU': 'C^inf', 'GELU': 'C^inf',
        'Softplus_10': 'C^inf', 'ReLU': 'C^0', 'Softplus_100': 'C^inf',
        'tanh': 'C^inf (sat.)', 'abs': 'C^0',
    }
    for act in act_order:
        row = [act]
        if act in expA.get('6', {}):
            row.append(f"{expA['6'][act]['leakage_ratio']:.4f}")
        else:
            row.append('-')
        if act in expC.get('6', {}) and 'GL_1x' in expC['6'][act]:
            row.append(f"{expC['6'][act]['GL_1x']['mean_equiv_error']:.4f}")
        else:
            row.append('-')
        row.append(smooth_label.get(act, ''))
        rows.append(row)
    pdf.add_table(headers, rows, col_widths=[40, 40, 45, 55])

    # ==================================================================
    # 3. EQUIVARIANCE ERROR VS. SAMPLING STRATEGY
    # ==================================================================
    pdf.add_page()
    pdf.section_title('3. Equivariance Error vs. Sampling Strategy')

    pdf.body_text(
        'We now hold the activation function fixed and vary the quadrature. '
        'A surprising dichotomy emerges: oversampling has almost no effect on '
        'coefficient accuracy, but dramatically reduces equivariance error. '
        'We also compare different quadrature families (Gauss-Legendre, Lebedev, '
        'and uniform grids).'
    )

    pdf.sub_title('3.1 The Truncation-Aliasing Dichotomy')
    pdf.body_text(
        'Figure 3 shows the relative coefficient error E_coeff as a function of '
        'GL oversampling ratio (1x, 2x, 3x) for each activation. The error is '
        'nearly flat: increasing the number of quadrature points from ~100 to ~700 '
        'barely changes the output coefficients.\n\n'
        'This is because coefficient error is dominated by truncation, not aliasing. '
        'When the nonlinearity generates energy at degrees l > l_max, that energy is '
        'lost regardless of how accurately we integrate -- the output SH basis simply '
        'cannot represent it. More quadrature points give a more accurate integral, '
        'but the integral is still projecting onto a truncated basis, so the '
        'irrecoverable truncation error dominates.'
    )

    # Fig 3: Coefficient error vs oversampling (= old B2)
    pdf.add_figure(
        'results/figures/expB_oversampling_decay.png',
        'Figure 3: Relative coefficient error E_coeff = ||c_out - c_gt|| / ||c_gt|| '
        'vs. GL oversampling ratio (1x, 2x, 3x) for each activation at l_max = 3, 6, 10. '
        'Ground truth: GL quadrature at l_max_ref = 25, truncated to l_max. '
        'The near-flat curves show that increasing quadrature resolution does not '
        'reduce coefficient error, because the error is dominated by truncation of '
        'high-l content, not by quadrature aliasing.'
    )

    pdf.body_text(
        'In contrast, Figure 4 shows the equivariance error E_equiv as a function '
        'of GL oversampling ratio. Here, oversampling produces dramatic improvements: '
        'GL_3x reduces equivariance error by 10x to 450x compared to GL_1x, '
        'depending on the activation.\n\n'
        'Why does oversampling help equivariance but not coefficient accuracy? '
        'The key insight is that truncation error is equivariant but aliasing error '
        'is not:\n\n'
        '  - Truncation is a projection: it discards all l > l_max components. '
        'This projection commutes with rotation (since each degree l transforms '
        'independently under SO(3)), so truncation alone does not break equivariance. '
        'It loses information, but it loses the same information regardless of '
        'orientation.\n\n'
        '  - Aliasing is quadrature-dependent: when the quadrature rule cannot '
        'exactly integrate products of high-l and low-l harmonics, high-frequency '
        'energy folds back into low-l coefficients. The folding pattern depends on '
        'the specific positions of quadrature points relative to the function being '
        'integrated. Since rotating the input moves the function relative to the '
        'fixed quadrature grid, the aliasing pattern changes with rotation, '
        'breaking equivariance.\n\n'
        'Oversampling reduces aliasing by providing enough quadrature points to '
        'accurately integrate the cross-terms between high-l and low-l harmonics. '
        'This is why smooth activations benefit more from oversampling: they have '
        'less high-l content to alias in the first place, so moderate oversampling '
        'can effectively eliminate aliasing for them.'
    )

    # Fig 4: Equivariance error vs oversampling (= old C2)
    pdf.add_figure(
        'results/figures/expC_equiv_vs_oversampling.png',
        'Figure 4: Equivariance error E_equiv vs. GL oversampling ratio (1x, 2x, 3x) '
        'for each activation at l_max = 3, 6, 10. Compare with Figure 3: while '
        'coefficient error is insensitive to oversampling, equivariance error drops '
        'by 1-2 orders of magnitude. This is because aliasing (not truncation) is the '
        'dominant source of equivariance violation, and oversampling directly reduces '
        'aliasing. Smooth activations (Softplus_1, SiLU) benefit most.'
    )

    # Equivariance error table for GL oversampling
    pdf.sub_title('Table 2: Equivariance Error vs. GL Oversampling (l_max = 6)')
    headers = ['Activation', 'GL_1x (N~98)', 'GL_2x (N~338)', 'GL_3x (N~722)',
               'Improvement']
    rows = []
    for act in act_order:
        if act in expC.get('6', {}):
            d = expC['6'][act]
            e1 = d.get('GL_1x', {}).get('mean_equiv_error', 0)
            e3 = d.get('GL_3x', {}).get('mean_equiv_error', 1)
            row = [act]
            for cfg in ['GL_1x', 'GL_2x', 'GL_3x']:
                if cfg in d:
                    row.append(f"{d[cfg]['mean_equiv_error']:.2e}")
                else:
                    row.append('-')
            row.append(f"{e1/e3:.0f}x" if e3 > 0 else '-')
            rows.append(row)
    pdf.add_table(headers, rows, col_widths=[30, 37, 37, 37, 49])

    pdf.sub_title('3.2 Comparison of Quadrature Methods')
    pdf.body_text(
        'Beyond oversampling within a single quadrature family, we compare three '
        'qualitatively different quadrature methods: Gauss-Legendre (GL), Lebedev, '
        'and uniform (equiangular) grids. Figure 5 shows the equivariance error '
        'for all six sampling configurations tested at l_max = 6.\n\n'
        'Key observations:\n'
        '  - At matched oversampling (GL_1x vs. Leb_min), GL slightly outperforms '
        'Lebedev, likely because GL nodes are optimally placed for the tensor-product '
        'structure of S^2.\n'
        '  - Lebedev at 2x (N = 230) performs comparably to GL_2x (N = 338) for '
        'smooth activations, achieving similar equivariance error with fewer points.\n'
        '  - Uniform grids at high resolution (N = 5000) achieve the lowest '
        'equivariance error across all activations, but at much higher cost. '
        'This confirms that the error reduction is a genuine quadrature effect, '
        'not an artifact of GL-specific structure.\n'
        '  - The relative ranking of activations is preserved across all quadrature '
        'methods: smoother activations consistently achieve lower equivariance error.'
    )

    # Fig 5: Sampling strategy comparison (NEW)
    pdf.add_figure(
        fig5_path,
        'Figure 5: Equivariance error E_equiv for each activation under six '
        'sampling strategies at l_max = 6. N denotes the number of quadrature points. '
        'GL_kx: Gauss-Legendre with k-times oversampling. '
        'Leb min/2x: Lebedev quadrature at minimum / double degree. '
        'Uniform: equiangular grid. Log scale. Error bars: std over 20 rotations '
        'x 10 inputs. The activation ranking is stable across methods, confirming '
        'that smoothness effects are independent of quadrature choice.'
    )

    # Sampling methods table
    pdf.sub_title('Table 3: Equivariance Error Across Sampling Methods (l_max = 6)')
    headers = ['Activation', 'GL_1x', 'GL_2x', 'Leb_min', 'Leb_2x', 'Uniform']
    cfg_keys = ['GL_1x', 'GL_2x', 'Leb_min_d13', 'Leb_2x_d25', 'Uniform_50']
    rows = []
    for act in ['Softplus_1', 'SiLU', 'ReLU', 'tanh', 'abs']:
        if act in expC.get('6', {}):
            d = expC['6'][act]
            row = [act]
            for cfg in cfg_keys:
                if cfg in d:
                    row.append(f"{d[cfg]['mean_equiv_error']:.2e}")
                else:
                    row.append('-')
            rows.append(row)
    n_row = ['N (points)']
    for cfg in cfg_keys:
        n_pts = expC['6']['Softplus_1'].get(cfg, {}).get('n_points', '?')
        n_row.append(str(n_pts))
    rows.insert(0, n_row)
    pdf.add_table(headers, rows, col_widths=[32, 32, 32, 32, 32, 32])

    # ==================================================================
    # 4. EXPRESSIBILITY AND TASK PERFORMANCE
    # ==================================================================
    pdf.add_page()
    pdf.section_title('4. Expressibility and Task Performance')

    pdf.sub_title('4.1 Measuring Expressibility: Effective Rank of the Jacobian')
    pdf.body_text(
        'Sections 2-3 characterize how activation choice and sampling affect '
        'equivariance error. However, equivariance is not the only property that '
        'matters for downstream performance: the activation must also be expressive '
        'enough to represent useful nonlinear transformations on the sphere. We now '
        'quantify this expressibility and examine whether it explains task performance '
        'differences that leakage alone cannot.\n\n'
        'Definition.  The S2 Activation operator A_sigma maps input SH coefficients '
        'c in R^d to output coefficients c\' in R^d (where d = (l_max+1)^2). '
        'At each input c, the Jacobian J(c) = dA_sigma/dc is a d x d matrix whose '
        'singular values {s_1 >= s_2 >= ... >= s_d} describe how the operator locally '
        'stretches or compresses each direction in coefficient space.\n\n'
        'We define the expressibility as the effective rank of J:\n'
        '    EffRank(J) = exp(H(p)),  where  p_i = s_i^2 / sum_j s_j^2\n'
        'and H(p) = -sum_i p_i log(p_i) is the Shannon entropy. The effective rank '
        'is a continuous quantity in [1, d] that counts how many independent directions '
        'in output space the S2 Activation can locally explore. A near-identity operator '
        '(all s_i equal) has EffRank = d; an operator that collapses most directions '
        '(one dominant s_i) has EffRank near 1.\n\n'
        'Computation.  For the S2 Activation forward pass c -> Y c -> sigma(.) -> '
        'wY^T (.), the Jacobian is J = wY^T diag(sigma\'(Yc)) Y, where Y is the '
        'SH evaluation matrix and wY the weighted inverse matrix. We compute J via '
        'autograd and take its SVD. Results are averaged over 50 random inputs '
        '(c_lm ~ N(0,1)) at each l_max.'
    )

    # Fig 6: Expressibility bar chart
    pdf.add_figure(
        'results/figures/expE_expressibility_bar.png',
        'Figure 6: Effective rank of the S2Activation Jacobian for each '
        'activation function at l_max = 3, 6, 10. Dashed line: maximum rank d = '
        '(l_max+1)^2. Higher effective rank means the operator explores more '
        'independent output directions. Red: C^0 activations. Green: Softplus family. '
        'Blue: C^inf smooth. Averaged over 50 random inputs.'
    )

    pdf.body_text(
        'The results reveal a surprising pattern that contradicts naive intuitions '
        'about expressibility:\n\n'
        '  - Softplus(1) has the HIGHEST effective rank despite being the smoothest '
        'activation. This is because smooth, non-saturating activations have well-'
        'conditioned derivatives across the sphere: sigma\'(f) is non-zero and varies '
        'smoothly, producing a full-rank Jacobian with evenly distributed singular '
        'values.\n\n'
        '  - tanh has surprisingly LOW effective rank, especially at high l_max '
        '(44.3/121 at l_max=10). This is because tanh saturates: for large |f(x)|, '
        'tanh\'(f) -> 0, collapsing those directions in output space. The larger l_max '
        'is, the more points on the sphere have large |f| values (due to higher '
        'variance), and the more directions are collapsed.\n\n'
        '  - abs has very HIGH effective rank because |sigma\'(f)| = 1 everywhere '
        '(except at f=0, a measure-zero set), preserving all directions despite '
        'being non-smooth.\n\n'
        '  - Softplus(beta) effective rank DECREASES with beta: Softplus(1) > '
        'Softplus(10) > Softplus(100) ~ ReLU. As beta increases, the activation '
        'approaches ReLU, which zeros out the derivative for all f < 0, collapsing '
        'roughly half the directions.'
    )

    pdf.sub_title('4.2 Expressibility vs. Spectral Leakage')
    pdf.body_text(
        'Figure 7 plots expressibility (effective rank) against spectral leakage '
        'for each activation. The two metrics are clearly separable -- they measure '
        'different properties of the activation:\n\n'
        '  - Softplus(1) sits in the HIGH expressibility / LOW leakage corner: it is '
        'both smooth (low spectral tails) and expressive (well-conditioned Jacobian).\n\n'
        '  - tanh sits in the LOW expressibility / MODERATE leakage region: its '
        'saturation collapses output directions, while its bounded range concentrates '
        'spectral energy.\n\n'
        '  - abs and ReLU sit in the HIGH leakage region but with different '
        'expressibility: abs preserves all directions (high rank) while ReLU zeros '
        'out half (moderate rank).\n\n'
        'This decomposition explains why leakage alone does not predict task '
        'performance: two activations with similar leakage can have very different '
        'expressibility, and vice versa.'
    )

    # Fig 7: Expressibility vs leakage scatter
    pdf.add_figure(
        'results/figures/expE_express_vs_leakage.png',
        'Figure 7: Effective rank (expressibility) vs. spectral leakage ratio '
        'for each activation at l_max = 3, 6, 10. Each point is one activation. '
        'The two axes measure orthogonal properties: leakage measures energy lost '
        'above l_max, while effective rank measures the diversity of output directions '
        'within the retained l_max subspace. Softplus(1) achieves both low leakage '
        'and high expressibility; tanh has low expressibility due to saturation.'
    )

    # Combined table
    pdf.sub_title('Table 4: Expressibility and Leakage (l_max = 6)')
    headers = ['Activation', 'Leakage R', 'EffRank', 'EffRank/d', 'Mechanism']
    rows = []
    mechanism = {
        'Softplus_1': 'Smooth, non-sat.', 'SiLU': 'Smooth, non-sat.',
        'GELU': 'Smooth, non-sat.', 'Softplus_10': 'Near-ReLU',
        'ReLU': 'C^0, half-zeroed', 'Softplus_100': 'Near-ReLU',
        'tanh': 'C^inf, saturating', 'abs': 'C^0, sign-flip',
    }
    for act in act_order:
        row = [act]
        if act in expA.get('6', {}):
            row.append(f"{expA['6'][act]['leakage_ratio']:.4f}")
        else:
            row.append('-')
        if act in expE.get('6', {}):
            er = expE['6'][act]['mean_effective_rank']
            row.append(f"{er:.1f}")
            row.append(f"{er/49:.2f}")
        else:
            row.append('-')
            row.append('-')
        row.append(mechanism.get(act, ''))
        rows.append(row)
    pdf.add_table(headers, rows, col_widths=[30, 30, 30, 30, 70])

    pdf.sub_title('4.3 Preliminary Task Performance (Experiment D)')
    pdf.body_text(
        'We train a 1-layer SphericalCNN on a synthetic 5-class classification task '
        '(l_max = 6), where class-discriminative information resides in high-l '
        'coefficients. 5 activations x 3 sampling configs, 2 runs each, 15 epochs.\n\n'
        'Results: Accuracy differences are small (1-2%). Neither leakage nor '
        'expressibility alone predicts the ranking. Notably:\n'
        '  - tanh (87.5%) has the best accuracy but LOW expressibility and MODERATE '
        'leakage.\n'
        '  - Softplus_1 (87.3%) is a close second with HIGH expressibility and LOW '
        'leakage.\n'
        '  - ReLU (86.3%) has MODERATE expressibility and HIGH leakage.\n\n'
        'The fact that tanh achieves high task accuracy despite low effective rank '
        'suggests that for a 1-layer model, the specific form of the nonlinearity on '
        'S^2 (saturation produces bounded, well-separated class responses) may matter '
        'more than local Jacobian diversity. In deeper models where equivariance '
        'errors compound, the ranking could change.'
    )

    pdf.sub_title('Table 5: Test Accuracy')
    headers = ['Activation', 'GL_1x', 'GL_2x', 'Leb_min']
    rows = []
    for act in ['tanh', 'Softplus_1', 'SiLU', 'ReLU', 'Softplus_10']:
        row = [act]
        for samp in ['GL_1x', 'GL_2x', 'Leb_min']:
            key = f"{act}_{samp}"
            if key in expD:
                row.append(f"{expD[key]['mean_test_acc']:.3f}")
            else:
                row.append('-')
        rows.append(row)
    pdf.add_table(headers, rows, col_widths=[45, 48, 48, 49])

    pdf.add_figure(
        'results/figures/expD_accuracy_comparison.png',
        'Figure 8: Test accuracy by (activation x sampling) configuration. '
        '2 runs per config, 15 epochs, l_max = 6. Error bars: std across runs.'
    )
    pdf.add_figure(
        'results/figures/expD_acc_vs_equiv.png',
        'Figure 9: Task accuracy vs. equivariance error. No clear monotonic trend. '
        'Combined with the expressibility results, this confirms that task performance '
        'depends on a complex interplay of leakage, expressibility, and model-specific '
        'factors that cannot be reduced to a single axis.',
        w=140
    )

    # ==================================================================
    # 5. CONCLUSIONS
    # ==================================================================
    pdf.add_page()
    pdf.section_title('5. Conclusions')
    pdf.body_text(
        'This study identifies three independent axes characterizing S2 Activation -- '
        'spectral leakage, aliasing sensitivity, and expressibility -- and reveals '
        'fundamental dichotomies in their interactions.\n\n'
        '1. ACTIVATION SMOOTHNESS CONTROLS EQUIVARIANCE VIA SPECTRAL LEAKAGE\n'
        '   Smoother activations (Softplus_1, SiLU) produce less spectral leakage, '
        'and leakage is strongly correlated with equivariance error (Figure 2). '
        'The Softplus(beta) family provides a continuous 1-parameter knob from smooth '
        '(beta = 1, low leakage) to sharp (beta -> inf ~ ReLU, high leakage). '
        'This relationship is consistent across all l_max values and sampling methods.\n\n'
        '2. THE TRUNCATION-ALIASING DICHOTOMY\n'
        '   Coefficient error is dominated by truncation (insensitive to oversampling, '
        'Figure 3), while equivariance error is dominated by aliasing (highly sensitive '
        'to oversampling, Figure 4). This is because truncation commutes with rotation '
        '(it projects out the same degrees regardless of orientation), but aliasing does '
        'not (the fold-back pattern depends on the function\'s orientation relative to '
        'the quadrature grid).\n\n'
        '3. QUADRATURE METHOD MATTERS LESS THAN OVERSAMPLING\n'
        '   GL, Lebedev, and uniform grids produce similar equivariance error at '
        'matched point counts (Figure 5). The dominant factor is how many points are '
        'used, not their specific arrangement.\n\n'
        '4. SMOOTHNESS AND EXPRESSIBILITY ARE NOT OPPOSED\n'
        '   The effective rank of the Jacobian (Figures 6-7) reveals that smooth '
        'activations are also the most expressive: Softplus(1) achieves both the lowest '
        'leakage and the highest effective rank. Saturation, not smoothness, reduces '
        'expressibility: tanh has low effective rank because tanh\'(x) -> 0 for large |x|. '
        'Activations that zero out half the input (ReLU, high-beta Softplus) also lose '
        'rank. The Pareto-optimal activation for both leakage and expressibility is '
        'Softplus(1) or SiLU.\n\n'
        '5. TASK PERFORMANCE REMAINS MULTI-FACTORIAL\n'
        '   Our preliminary Experiment D shows that neither leakage nor expressibility '
        'alone predicts task accuracy: tanh achieves the best accuracy despite low '
        'expressibility. In a 1-layer model, the global shape of the nonlinearity '
        '(saturation, boundedness) likely matters more than local Jacobian diversity. '
        'In deeper models where equivariance errors compound, the ranking could shift.\n\n'
        '6. PRACTICAL GUIDANCE (for equivariance)\n'
        '   - To minimize equivariance error: use Softplus(1) or SiLU + >= 2x GL '
        'oversampling.\n'
        '   - Cost-effective default: SiLU + GL_1x (standard in most models).\n'
        '   - Parametric control: Softplus(beta) to trade smoothness vs. sharpness.\n'
        '   - Caveat: minimizing equivariance error does not guarantee best task '
        'performance in shallow models.'
    )

    # ==================================================================
    # APPENDIX
    # ==================================================================
    pdf.add_page()
    pdf.section_title('Appendix A: Additional Spectral Leakage Figures')
    pdf.body_text(
        'These figures provide additional detail on the spectral structure of '
        'different activations beyond the summary in Figure 1.'
    )

    pdf.add_figure(
        'results/figures/expA_power_spectra.png',
        'Figure A1: Full power spectrum P(l) vs. degree l after applying each '
        'nonlinearity to a random band-limited input (l <= l_max). Dashed vertical '
        'line marks l_max. Energy to the right of this line is lost upon truncation. '
        'Panels: l_max = 3, 6, 10. Log scale.'
    )
    pdf.add_figure(
        'results/figures/expA_softplus_spectra.png',
        'Figure A2: Power spectra for the Softplus(beta) family with ReLU reference '
        '(dashed red). As beta increases, the spectrum approaches the ReLU power-law '
        'tail. Panels: l_max = 3, 6, 10.'
    )
    pdf.add_figure(
        'results/figures/expA_softplus_transition.png',
        'Figure A3: Leakage ratio R vs. Softplus beta on a log scale, for '
        'l_max = 3, 6, 10. Horizontal dashed lines show the corresponding ReLU '
        'leakage. Confirms continuous interpolation from smooth to sharp.',
        w=130
    )

    # Leakage table (full)
    pdf.sub_title('Table A1: Leakage Ratio R (Full)')
    acts_order = ['Softplus_1', 'Softplus_3', 'SiLU', 'GELU', 'LeakyReLU_0.1',
                  'ReLU', 'Softplus_10', 'Softplus_30', 'Softplus_100',
                  'tanh', 'sin', 'abs', 'x^2']
    headers = ['Activation', 'R (l_max=3)', 'R (l_max=6)', 'R (l_max=10)']
    rows = []
    for act in acts_order:
        row = [act]
        for lmax in ['3', '6', '10']:
            if act in expA.get(lmax, {}):
                row.append(f"{expA[lmax][act]['leakage_ratio']:.4f}")
            else:
                row.append('-')
        rows.append(row)
    pdf.add_table(headers, rows, col_widths=[45, 48, 48, 49])

    pdf.add_page()
    pdf.section_title('Appendix B: Additional Coefficient Error Figures')

    pdf.add_figure(
        'results/figures/expB_error_vs_npoints.png',
        'Figure B1: Coefficient error vs. number of sampling points for each '
        'activation. Even at N = 20000, error plateaus at the truncation floor. '
        'Only smoother activations achieve lower error.'
    )
    pdf.add_figure(
        'results/figures/expB_heatmap_lmax6.png',
        'Figure B2: Coefficient error heatmap (activation x sampling method) at '
        'l_max = 6. Rows: activations. Columns: sampling configurations. Color: '
        'relative error. Confirms that the activation axis drives error, not sampling.',
        w=150
    )

    # Coefficient error table
    pdf.sub_title('Table B1: Coefficient Error (l_max = 6)')
    headers = ['Activation', 'GL_1x', 'GL_2x', 'GL_3x', 'Trunc Ratio']
    rows = []
    for act in ['Softplus_1', 'SiLU', 'GELU', 'ReLU', 'Softplus_10', 'tanh',
                'abs', 'Softplus_100']:
        if act in expB.get('6', {}):
            d = expB['6'][act]
            trunc = d.get('GL_1x', {}).get('mean_trunc_ratio', 0)
            row = [act]
            for cfg in ['GL_1x', 'GL_2x', 'GL_3x']:
                if cfg in d:
                    row.append(f"{d[cfg]['mean_rel_error']:.3f}")
                else:
                    row.append('-')
            row.append(f"{trunc:.4f}")
            rows.append(row)
    pdf.add_table(headers, rows, col_widths=[35, 35, 35, 35, 50])

    pdf.add_page()
    pdf.section_title('Appendix C: Additional Equivariance Figures')

    pdf.add_figure(
        'results/figures/expC_equiv_vs_activation.png',
        'Figure C1: Equivariance error grouped by activation, with separate bars '
        'for GL_1x, GL_2x, GL_3x. Panels: l_max = 3, 6, 10.'
    )

    pdf.add_page()
    pdf.section_title('Appendix D: Phase 1 - Quadrature Method Comparison')
    pdf.body_text(
        'Phase 1 compared quadrature methods (GL, Lebedev, Uniform, Fibonacci) for '
        'pure SH reconstruction accuracy without nonlinearities.'
    )
    pdf.add_figure('results/figures/exp1_accuracy_curves.png',
                   'Figure D1: SH reconstruction error vs sampling points')
    pdf.add_figure('results/figures/exp1_error_by_degree.png',
                   'Figure D2: Per-degree reconstruction error')
    pdf.add_figure('results/figures/exp2_time_scaling.png',
                   'Figure D3: Computational cost scaling', w=140)
    pdf.add_figure('results/figures/exp4_efficiency_frontier.png',
                   'Figure D4: Efficiency frontier')
    pdf.add_figure('results/figures/exp4_asymptotic.png',
                   'Figure D5: Asymptotic complexity', w=130)

    # Save
    pdf.output('report.pdf')
    size_kb = os.path.getsize('report.pdf') / 1024
    print(f'PDF saved: report.pdf ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
