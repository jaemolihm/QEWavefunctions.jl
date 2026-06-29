#!/usr/bin/env python3

import sys
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_nk(filename):
    with open(filename, 'r') as f:
        nk = 0
        for line in f:
            if line[0] == "#": continue
            if line.strip() == "":
                break
            nk += 1
    return nk

def parse_efermi_or_evbm(filename):
    # Return the last found value because when using hybrid functions, Fermi energy is
    # printed multiple times and the last one is the one for the converged bands.
    out = None
    with open(filename, 'r') as f:
        for line in f:
            if "highest occupied level" in line:
                out = float(line.split()[-1]), None
            if "the Fermi energy is" in line:
                out = float(line.split()[-2]), None
            if "highest occupied, lowest unoccupied level" in line:
                out = float(line.split()[-2]), float(line.split()[-1])
    return out

def merge_multiple_high_sym_labels(high_sym_k, high_sym_label):
    high_sym_k_new = [high_sym_k[0]]
    high_sym_label_new = [high_sym_label[0]]
    for i in range(1, len(high_sym_k)):
        if abs(high_sym_k[i] - high_sym_k_new[-1]) > 1e-5:
            # label at different x
            high_sym_k_new += [high_sym_k[i]]
            high_sym_label_new += [high_sym_label[i]]
        else:
            # label at same x
            high_sym_label_new[-1] += "|" + high_sym_label[i]
    return high_sym_k_new, high_sym_label_new


# Parse input arguments
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--pdf", action='store_true')
    parser.add_argument("--png", action='store_true')
    parser.add_argument('--ymin', help="Energy below Fermi level to show", nargs='?', const=1, default=-2.0, type=float)
    parser.add_argument('--ymax', help="Energy above Fermi level to show", nargs='?', const=1, default=2.0, type=float)
    args = parser.parse_args()
    prefix = args.prefix
    write_pdf = args.pdf
    write_png = args.png
    ymin = args.ymin
    ymax = args.ymax

    filename_w90 = f"{prefix}_band.dat"
    filename_pw = f"{prefix}.bands.dat.gnu"
    filename_label = f"{prefix}_band.labelinfo.dat"
    filename_label_pw = f"plotband.out"

    # SCF filename is "scf.out" or "{prefix}.scf.out"
    filename_scf = "scf.out"
    if not os.path.isfile(filename_scf):
        if os.path.isfile(f"{prefix}.scf.out"):
            filename_scf = f"{prefix}.scf.out"
        else:
            filename_scf = None

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Wannier90 bands
    if os.path.isfile(filename_w90):
        nk_w90 = get_nk(filename_w90)

        data_w90 = np.loadtxt(filename_w90).reshape((-1, nk_w90, 2))
        xkplot_w90 = data_w90[0, :, 0]
        e_w90 = data_w90[:, :, 1]

        for ax in axes:
            for ib in range(e_w90.shape[0]):
                lines = ax.plot(xkplot_w90, e_w90[ib, :], 'r--', label='W90' if ib==0 else None, zorder=3, lw=1)
            ax.set_xlim([min(xkplot_w90), max(xkplot_w90)])
    else:
        xkplot_w90 = [1.0]

    # DFT bands
    if os.path.isfile(filename_pw):
        # Read LSDA information if present.
        with open(filename_pw, "r") as f:
            line = f.readline()
            if "# LSDA" in line:
                lsda = True
                nbnd_up, nbnd_dw = [int(x) for x in line.split()[-2:]]
            else:
                lsda = False

        nk_pw = get_nk(filename_pw)
        data_pw = np.loadtxt(filename_pw).reshape((-1, nk_pw, 2))
        xkplot_pw = data_pw[0, :, 0]
        e_pw = data_pw[:, :, 1]
        xk_pw_to_w90_convert = xkplot_w90[-1] / xkplot_pw[-1]
        xkplot_pw *= xk_pw_to_w90_convert
        nbnd = e_pw.shape[0]

        for ax in axes:
            if lsda:
                for ib in range(nbnd_up):
                    lines = ax.plot(xkplot_pw, e_pw[ib, :], 'k-', label='DFT spin up' if ib==0 else None, zorder=1)
                for ib in range(nbnd_up, nbnd_up + nbnd_dw):
                    lines = ax.plot(xkplot_pw, e_pw[ib, :], 'b-', label='DFT spin down' if ib==nbnd_up else None, zorder=1)
            else:
                for ib in range(nbnd):
                    lines = ax.plot(xkplot_pw, e_pw[ib, :], 'k-', label='DFT' if ib==0 else None, zorder=1)
            ax.set_xlim([min(xkplot_pw), max(xkplot_pw)])

    # Special k points
    high_sym_k = []
    high_sym_label = []

    # Try to append prefix to filename_label_pw
    if (not os.path.isfile(filename_label_pw)) and os.path.isfile(f"{prefix}.{filename_label_pw}"):
        filename_label_pw = f"{prefix}.{filename_label_pw}"

    if os.path.isfile(filename_label):
        # Read high-symmetry k point labels from Wannier90 output
        with open(filename_label, 'r') as f:
            for line in f:
                if line.strip() == "":
                    break
                data = line.split()
                high_sym_label += [data[0]]
                high_sym_k += [float(data[2])]
    elif os.path.isfile(filename_label_pw):
        # Read high-symmetry k point labels from my_qe_bands.py output
        with open(filename_label_pw, 'r') as f:
            for line in f:
                if 'high-symmetry point:' in line:
                    high_sym_k += [float(line.split()[7])]
                    if len(line.split()) == 9:
                        high_sym_label += [line.split()[8]]
                    else:
                        high_sym_label += ['.']
        high_sym_k = [x * xk_pw_to_w90_convert for x in high_sym_k]
    else:
        # Skip high-symmetry k points
        pass
    high_sym_label = ["$\Gamma$" if x.lower() in ['gamma', 'g'] else x for x in high_sym_label]
    high_sym_label = ["$\Sigma$" if x.lower() == 'sigma' else x for x in high_sym_label]

    # If there are multiple high-symmetry labels on the same k, merge them
    if high_sym_label:
        high_sym_k, high_sym_label = merge_multiple_high_sym_labels(high_sym_k, high_sym_label)

    if len(high_sym_k) > 0:
        for ax in axes:
            ax.set_xticks(high_sym_k)
            ax.set_xticklabels(high_sym_label)
            for x in high_sym_k:
                ax.axvline(x=x, c='k', lw=1, ls='--')

    # Fermi level
    if os.path.isfile(filename_scf):
        # efermi1: E_VBM (insulator) or Fermi energy (metal)
        # efermi2: E_CBM (insulator) or None (metal)
        efermi1, efermi2 = parse_efermi_or_evbm(filename_scf)
        if efermi1 is not None:
            for ax in axes:
                ax.axhline(y=efermi1, c='b', lw=1)
            axes[1].set_ylim([efermi1 + ymin, efermi1 + ymax])
        if efermi2 is not None:
            for ax in axes:
                ax.axhline(y=efermi2, c='g', lw=1)

    axes[0].set_ylabel("Energy (eV)")
    axes[0].legend()
    if write_pdf:
        print(f"Writing PDF : {prefix}_band.pdf")
        plt.savefig(f"{prefix}_band.pdf")
    if write_png:
        print(f"Writing PNG : {prefix}_band.png")
        plt.savefig(f"{prefix}_band.png", dpi=300)
    plt.show()
