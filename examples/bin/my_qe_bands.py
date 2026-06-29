#!/usr/bin/env python3
"""
This script mimics what the QE bands.x does when lsym=.false. is set,
by parsing the seedname.xml file in the outdir.
1) print out the "high-symmetry points" by looking at the kinks in the
k-point path, and
2) writes a "seedname.bands.dat.gnu" file that contains band structure
information directly plottable by gnuplot.

Usage: python3 ~/bin/my_qe_bands.py seedname outdir > seedname.plotband.out
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

hartree_to_ev = lambda x: float(x) * 27.21138602
bohr_to_ang = lambda x: float(x) * 0.529177249

def parse_bandsdata(xml_file):
    from copy import copy

    print("Reading {}".format(xml_file))
    bandsdata = {}

    # parse xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    alat = float(root.find('.//atomic_structure').get('alat'))
    ecutwfc = hartree_to_ev(root.find('.//ecutwfc').text)
    ecutrho = hartree_to_ev(root.find('.//ecutrho').text)
    lsda = root.find('.//lsda').text.strip().lower() == "true"
    nelec = int(round(float(root.find('.//nelec').text)))
    if lsda:
        nbnd_up = int(round(float(root.find('.//nbnd_up').text)))
        nbnd_dw = int(round(float(root.find('.//nbnd_dw').text)))
        nbnd = nbnd_up + nbnd_dw
        bandsdata['nbnd_up'] = nbnd_up
        bandsdata['nbnd_dw'] = nbnd_dw
        bandsdata['nbnd'] = nbnd
    else:
        nbnd = int(round(float(root.find('.//nbnd').text)))
        bandsdata['nbnd'] = nbnd
    nks = int(round(float(root.find('.//nks').text)))
    bandsdata['alat'] = alat
    bandsdata['ecutwfc'] = ecutwfc
    bandsdata['ecutrho'] = ecutrho
    bandsdata['dual'] = ecutrho / ecutwfc
    bandsdata['nelec'] = nelec
    bandsdata['nks'] = nks
    bandsdata['lsda'] = lsda

    nspin = 2 if lsda else 1

    node_band = root.find('.//band_structure')
    band_energies = np.zeros((nks, nbnd))
    occupations = np.zeros((nks, nbnd))
    iks = 0
    for node_ks in node_band.findall('ks_energies'):
        band_energies[iks,:] = [hartree_to_ev(x) for x in node_ks.find('eigenvalues').text.split()]
        occupations[iks,:] = [float(x) for x in node_ks.find('occupations').text.split()]
        iks += 1
    bandsdata['band_energies'] = band_energies.copy()
    bandsdata['occupations'] = occupations.copy()

    kpoints_cart = np.zeros((3, nks))
    node_kpoint = root.find('.//starting_k_points')
    iks = 0
    for node_ks in node_kpoint.findall('k_point'):
        kpoints_cart[:,iks] = [float(x) for x in node_ks.text.split()]
        iks += 1
    bandsdata['kpoints_cart'] = kpoints_cart.copy()
    return bandsdata

def punch_plottable_bands(bandsdata, gnu_file):
    """
    Ported from the fortran subroutine punch_plottable_bands in bands.f90 of
    Quantum ESPRESSO v6.4
    """
    MAX_LINES = 100

    nks = bandsdata['nks']
    nbnd = bandsdata['nbnd']
    kpts = bandsdata['kpoints_cart']
    high_symmetry = [False] * nks
    for ik in range(nks):
        if ik==0 or ik==nks-1:
            high_symmetry[ik] = True
        else:
            k1 = kpts[:,ik] - kpts[:,ik-1]
            k2 = kpts[:,ik+1] - kpts[:,ik]
            if np.linalg.norm(k1) < 1.E-4 or np.linalg.norm(k2) < 1.E-4:
                sys.stdout.write('punch_plottable_bands: two consecutive same k, exiting\n')
                raise ValueError()
            ps = np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2))
            high_symmetry[ik] = abs(ps-1.0) > 1.E-4 # If kink, then high symmetry

            # The gamma point is a high symmetry point
            if np.linalg.norm(kpts[:,ik]) < 1.E-4:
                high_symmetry[ik] = True

            # save the typical length of dk
            if ik == 1:
                dxmod_save = np.linalg.norm(k1)

    kx_plot = np.zeros(nks)
    for ik in range(1, nks):
        dxmod = np.linalg.norm(kpts[:,ik] - kpts[:,ik-1])
        if dxmod > 5*dxmod_save:
            # A big jump in dxmod is a sign that the point xk(:,n) and xk(:,n-1)
            # are quite distant and belong to two different lines. We put them on
            # the same point in the graph
            kx_plot[ik] = kx_plot[ik-1]
        elif dxmod > 1.E-4:
            # This is the usual case. The two points xk(:,n) and xk(:,n-1) are in
            # the same path.
            kx_plot[ik] = kx_plot[ik-1] + dxmod
            dxmod_save = dxmod
        else:
            # This is the case in which dxmod is almost zero. The two points
            # coincide in the graph, but we do not save dxmod.
            kx_plot[ik] = kx_plot[ik-1] + dxmod

    # Now we compute how many paths there are: nlines
    # The first point of this path: point(iline)
    # How many points are in each path: npoints(iline)
    npoints = np.zeros(nks)
    point = np.zeros(nks)
    for ik in range(nks):
        if high_symmetry[ik]:
            if ik == 0:
                # first point. Initialize the number of lines, and the number of point
                # and say that this line start at the first point
                nlines = 1
                npoints[0] = 1
                point[0] = 1
            elif ik == nks-1:
                # Last point. Here we save the last point of this line, but
                # do not increase the number of lines
                npoints[nlines-1] = npoints[nlines-1] + 1
                point[nlines] = ik + 1
            else:
                # Middle line. The current line has one more points, and there is
                # a new line that has to be initialized. It has one point and its
                # first point is the current k.
                npoints[nlines-1] = npoints[nlines-1] + 1
                nlines = nlines + 1
                if nlines > MAX_LINES:
                    sys.stdout.write('punch_plottable_bands: too many lines, exiting\n')
                    raise ValueError()
                npoints[nlines-1] = 1
                point[nlines-1] = ik + 1
            sys.stdout.write("high-symmetry point: {0:7.4f} {1:7.4f} {2:7.4f}".format(kpts[0,ik], kpts[1,ik], kpts[2,ik]))
            sys.stdout.write(" x coordinate {0:9.4f}\n".format(kx_plot[ik]))
        else:
            # This k is not an high symmetry line so we just increase the number
            # of points of this line.
            npoints[nlines-1] = npoints[nlines-1] + 1

    with open(gnu_file, 'w') as f:
        if bandsdata["lsda"]:
            f.write(f'# LSDA : nbnd_up, nbnd_dw = {bandsdata["nbnd_up"]} {bandsdata["nbnd_dw"]}\n')
        for ib in range(nbnd):
            for ik in range(nks):
                f.write("{0:10.4f} {1:10.4f}\n".format(kx_plot[ik], bandsdata['band_energies'][ik,ib]))
            f.write('\n')

    with open('xkplot.txt', 'w') as f:
        for ik in range(nks):
            f.write(f"{kx_plot[ik]:12.7f}\n")

    return



if __name__ == "__main__":
    from time import localtime, strftime
    # read the input file
    header = ( 'Program my_qe_bands.py at '
              + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()) + '\n')
    sys.stdout.write(header)
    try:
        prefix = sys.argv[1]
        outdir = sys.argv[2]
    except IndexError:
        sys.stdout.write("Input file is not given (The standard bands.x input file)\n")
        raise IndexError()
    if outdir[-1] != '/':
        outdir += '/'
    filband = f"{prefix}.bands.dat"

    try:
        sys.stdout.write('prefix  : {}\n'.format(prefix))
        sys.stdout.write('outdir  : {}\n'.format(outdir))
        sys.stdout.write('filband : {}\n'.format(filband))
    except NameError:
        sys.stdout.write("Error in reading input file: standard bands.x input format is needed\n")
        raise NameError()

    # parse xml file
    #xml_file = os.path.join(outdir, prefix+'.xml')
    xml_file = os.path.join(outdir, prefix + '.save', 'data-file-schema.xml')
    print(xml_file)
    if not os.path.exists(xml_file):
        sys.stdout.write("Error in xml file: file does not exist\n")
        raise ValueError()
    try:
        bandsdata = parse_bandsdata(xml_file)
    except:
        sys.stdout.write("Error in reading xml file\n")
        raise ValueError()

    # Write high-symmetry k-points: the "poor man's algorithm" in bands.x
    gnu_file = os.path.join(filband+'.gnu')
    punch_plottable_bands(bandsdata, gnu_file)
    sys.stdout.write("Plottable bands (eV) written to file {}\n".format(filband+'.gnu'))

