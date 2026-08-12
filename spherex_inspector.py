#!/usr/bin/env python
"""
spherex_inspector.py
=====================
Rapid visual-inspection GUI for SPHEREx spectra (0.73-5.02 micron),
aimed at identifying high-z quasars via Halpha (and other) emission lines.

Courtesy of Claude Sonnet 5.

USAGE
-----
    python spherex_inspector.py <name> [--start_id N] [--radius 5]

Expects photometry files at:
    results_<name>/<id>_spherex_photometry.csv
with (at least) columns: wv_um, opt_flux_uJy, opt_flux_uJy_err
optionally: detector_id, ra, dec

Produces / resumes a persistent catalog:
    results_<name>/<name>_inspection_catalog.csv

DEPENDENCIES
------------
    numpy, pandas, matplotlib, astropy   (required)
    astroquery                            (optional - for SIMBAD/NED queries)

KEYBINDINGS  (see also the live side-panel in the GUI)
-----------
  Mouse click        set redshift using the wavelength & currently active line
  1 / 2 / 3 / 4 / 5   set active line = Halpha / Hbeta / MgII2800 / Pabeta / Paalpha
  0                   clear current redshift
  z                   type a redshift manually (terminal prompt)

  up / down           zoom y-axis in / out
  shift+up/down       pan y-axis up / down
  a                   autoscale y-axis (reset)
  (scroll wheel)      zoom y-axis around cursor

  n / right arrow     next spectrum
  b / left arrow      previous spectrum
  j                   jump to a specific id (terminal prompt)

  s                   save current z / line as "candidate" (does not advance)
  x                   flag as STAR / contaminant, save, and advance
  g                   flag as LOW-Z GALAXY, save, and advance
  u                   flag as UNKNOWN / unclassifiable, save, and advance
  c                   add a free-text comment (terminal prompt)

  i                   query SIMBAD (astroquery, prints result + shows match)
  d                   query NED    (astroquery, prints result + shows match)
  v                   open SIMBAD web page for this position in browser
  w                   open NED web page for this position in browser
  k                   manually enter RA/Dec (if not found in the CSV)

  h                   print full help to the terminal
"""

import os
import sys
import glob
import argparse
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from IPython import embed

# ----------------------------------------------------------------------
# Disable matplotlib's default keymaps that would otherwise collide with
# our shortcuts (s, h, g, k, a, v, q, left/right, etc.)
# ----------------------------------------------------------------------
for _key in ['keymap.fullscreen', 'keymap.home', 'keymap.back', 'keymap.forward',
             'keymap.pan', 'keymap.zoom', 'keymap.save', 'keymap.quit',
             'keymap.grid', 'keymap.grid_minor', 'keymap.yscale',
             'keymap.xscale']:#, 'keymap.all_axes']:
    matplotlib.rcParams[_key] = []

# ----------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False

HAVE_SIMBAD = False
try:
    from astroquery.simbad import Simbad
    HAVE_SIMBAD = True
except ImportError:
    pass

HAVE_NED = False
try:
    from astroquery.ipac.ned import Ned
    HAVE_NED = True
except ImportError:
    try:
        from astroquery.ned import Ned
        HAVE_NED = True
    except ImportError:
        pass

# ----------------------------------------------------------------------
# Rest-frame wavelengths of interest (micron)
# ----------------------------------------------------------------------
REST_WAVE = {
    'Lya':  0.121567,   # Lyman-alpha, 1215.67 A (vacuum)
    'MgII': 0.279994,   # Mg II 2796.35/2803.53 doublet average (vacuum)
    'Hg':   0.434169,   # H-gamma, 4341.69 A (vacuum)
    'Hb':   0.486269,   # H-beta, 4862.69 A (vacuum)
    'OIII': 0.500824,   # [O III] 5008.24 A (vacuum)
    'Ha':   0.656461,   # H-alpha, 6564.61 A (vacuum)
    'Pag':  1.094108,   # Paschen-gamma, 10941.08 A (vacuum)
    'Pab':  1.282158,   # Paschen-beta, 12821.58 A (vacuum)
    'Paa':  1.875610,   # Paschen-alpha, 18756.10 A (vacuum)
}

LINE_COLORS = {
    'Lya': 'purple', 'MgII': 'darkorange', 'Hg': 'seagreen', 'Hb': 'seagreen',
    'OIII': 'teal', 'Ha': 'crimson', 'Pag': 'royalblue', 'Pab': 'royalblue',
    'Paa': 'royalblue',
}

ACTIVE_LINE_KEYS = {'1': 'Ha', '2': 'Hb', '3': 'MgII', '4': 'Pab', '5': 'Paa'}


class SpherexInspector:
    def __init__(self, name, results_dir=None, outdir=None,
                 start_id=None, radius_arcsec=5.0):
        self.name = name
        self.results_dir = results_dir or f"results_{name}"
        self.outdir = outdir or self.results_dir
        self.radius = radius_arcsec

        self.ids = self._discover_ids()
        if not self.ids:
            raise RuntimeError(f"No spectra found in {self.results_dir}")

        self.catalog_path = os.path.join(
            self.outdir, f"{name}_inspection_catalog.csv")
        self.catalog = self._load_catalog()

        self.idx = 0
        if start_id is not None and start_id in self.ids:
            self.idx = self.ids.index(start_id)
        else:
            self.idx = self._first_unfinished_index()

        self.active_line = 'Ha'
        self.z = None
        self.ra = None
        self.dec = None
        self.line_artists = []
        self.spec = None
        self.pending_comment = ''
        self.last_simbad_match = '-'
        self.last_ned_match = '-'

        self._build_figure()
        self.load_current()
        plt.show()

    # ------------------------------------------------------------------
    # discovery / catalog I/O
    # ------------------------------------------------------------------
    def _discover_ids(self):
        files = glob.glob(os.path.join(self.results_dir,
                                        "*_spherex_photometry.csv"))
        ids = []
        for f in files:
            base = os.path.basename(f)
            try:
                ids.append(int(base.split("_")[0]))
            except ValueError:
                pass
        return sorted(ids)

    def _load_catalog(self):
        cols = ['id', 'z', 'line', 'ra', 'dec', 'classification', 'comment',
                'simbad_match', 'ned_match', 'timestamp']
        if os.path.exists(self.catalog_path):
            df = pd.read_csv(self.catalog_path)
            return df.set_index('id', drop=False)
        return pd.DataFrame(columns=cols).set_index('id', drop=False)

    def _save_catalog(self):
        self.catalog.sort_index().to_csv(self.catalog_path, index=False)

    def _first_unfinished_index(self):
        done = set(self.catalog['id'].tolist()) if len(self.catalog) else set()
        for i, oid in enumerate(self.ids):
            if oid not in done:
                return i
        return 0

    # ------------------------------------------------------------------
    # figure construction
    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 4, width_ratios=[3, 3, 3, 1.4])
        self.ax = self.fig.add_subplot(gs[0, :3])
        self.info_ax = self.fig.add_subplot(gs[0, 3])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(
            0, 1, '', va='top', ha='left', fontsize=8.5,
            family='monospace', transform=self.info_ax.transAxes)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.tight_layout()

    # ------------------------------------------------------------------
    # spectrum loading / plotting
    # ------------------------------------------------------------------
    def load_current(self):
        while True:
            oid = self.ids[self.idx]
            path = os.path.join(self.results_dir,
                                 f"{oid}_spherex_photometry.csv")
            try:
                self.spec = pd.read_csv(path)
                break
            except Exception as e:
                print(f"[WARN] failed to load id={oid}: {e}; skipping.")
                if self.idx < len(self.ids) - 1:
                    self.idx += 1
                else:
                    raise RuntimeError("No more spectra to load.")

        self.current_id = oid
        self.z = None
        self.active_line = 'Ha'
        self.pending_comment = ''
        self.last_simbad_match = '-'
        self.last_ned_match = '-'
        self.ra, self.dec = self._get_coords()

        if oid in self.catalog.index:
            row = self.catalog.loc[oid]
            if not pd.isna(row.get('z', np.nan)):
                self.z = float(row['z'])
            if isinstance(row.get('line'), str) and row['line']:
                self.active_line = row['line']

        self.plot_spectrum()
        self.redraw_lines()
        self.update_info_panel()
        self.fig.canvas.draw_idle()

    def _get_coords(self):
        for rc, dc in [('ra', 'dec'), ('RA', 'DEC'), ('input_ra_deg', 'input_dec_deg')]:
            if rc in self.spec.columns and dc in self.spec.columns:
                try:
                    return float(self.spec[rc].iloc[0]), float(self.spec[dc].iloc[0])
                except Exception:
                    pass
        master = os.path.join(self.results_dir, f"{self.name}_catalog.csv")
        if os.path.exists(master):
            try:
                mdf = pd.read_csv(master)
                if 'id' in mdf.columns and self.current_id in mdf['id'].values:
                    row = mdf[mdf['id'] == self.current_id].iloc[0]
                    if 'ra' in row and 'dec' in row:
                        return float(row['ra']), float(row['dec'])
            except Exception:
                pass
        return None, None

    def plot_spectrum(self):
        self.ax.cla()
        wv = self.spec['wv_um'].values
        flux = self.spec['opt_flux_uJy'].values / 1000.0
        err = self.spec['opt_flux_uJy_err'].values / 1000.0
        det = (self.spec['detector_id'].values
               if 'detector_id' in self.spec.columns else np.zeros_like(wv))

        self.ax.errorbar(wv, flux, yerr=err, c='k', markersize=3, fmt='o',
                          elinewidth=1, zorder=5)
        self.ax.scatter(wv, flux, c=det, marker='o', s=20, cmap='rainbow',
                         zorder=10)

        self.ax.set_xlabel('Observed wavelength (micron)')
        self.ax.set_ylabel('Flux (mJy)')
        self.ax.set_xlim(0.7, 5.1)

        med = np.nanmedian(flux) if len(flux) else 1.0
        self.default_ylim = (-0.1, 4 * med if med > 0 else 1.0)
        self.ax.set_ylim(*self.default_ylim)
        self.ax.set_title(f"id={self.current_id}  ({self.idx + 1}/{len(self.ids)})")
        self.fig.tight_layout()

    def redraw_lines(self):
        for art in self.line_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.line_artists = []

        if self.z is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        for name, rest in REST_WAVE.items():
            obs = rest * (1 + self.z)
            if obs < xlim[0] or obs > xlim[1]:
                continue
            is_active = (name == self.active_line)
            vline = self.ax.axvline(obs, color=LINE_COLORS.get(name, 'gray'),
                                     lw=2 if is_active else 1,
                                     ls='-' if is_active else '--',
                                     alpha=0.85, zorder=1)
            txt = self.ax.text(obs, ylim[1] * 0.97, name, rotation=90,
                                va='top', ha='right', fontsize=8,
                                color=LINE_COLORS.get(name, 'gray'))
            self.line_artists += [vline, txt]

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        rest = REST_WAVE[self.active_line]
        self.z = event.xdata / rest - 1.0
        self.redraw_lines()
        self.update_info_panel()
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.ydata is None:
            return
        base_scale = 1.2
        y0, y1 = self.ax.get_ylim()
        yc = event.ydata
        factor = (1 / base_scale) if event.button == 'up' else base_scale
        half = (y1 - y0) * factor / 2.0
        self.ax.set_ylim(yc - half, yc + half)
        self.redraw_lines()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key

        if k in ACTIVE_LINE_KEYS:
            self.active_line = ACTIVE_LINE_KEYS[k]
            self.redraw_lines()
            self.update_info_panel()
            self.fig.canvas.draw_idle()
            return

        if k == '0':
            self.z = None
            self.redraw_lines(); self.update_info_panel()
            self.fig.canvas.draw_idle(); return

        if k == 'z':
            self._prompt_manual_redshift(); return

        if k == 'up':
            self._zoom_y(0.8); return
        if k == 'down':
            self._zoom_y(1.25); return
        if k == 'shift+up':
            self._shift_y(0.2); return
        if k == 'shift+down':
            self._shift_y(-0.2); return
        if k == 'a':
            self.ax.set_ylim(*self.default_ylim)
            self.redraw_lines(); self.fig.canvas.draw_idle(); return

        if k in ('right', 'n'):
            self.next_spectrum(); return
        if k in ('left', 'b'):
            self.prev_spectrum(); return
        if k == 'j':
            self._prompt_jump(); return

        if k == 's':
            self.save_current(classification='candidate'); return
        if k == 'x':
            self.save_current(classification='star'); self.next_spectrum(); return
        if k == 'g':
            self.save_current(classification='low-z galaxy'); self.next_spectrum(); return
        if k == 'u':
            self.save_current(classification='unknown'); self.next_spectrum(); return
        if k == 'c':
            self._prompt_comment(); return

        if k == 'i':
            self.query_simbad(); return
        if k == 'd':
            self.query_ned(); return
        if k == 'v':
            self.open_simbad_browser(); return
        if k == 'w':
            self.open_ned_browser(); return
        if k == 'k':
            self._prompt_manual_coords(); return

        if k == 'h':
            self.print_help(); return

    # ------------------------------------------------------------------
    # y-axis helpers
    # ------------------------------------------------------------------
    def _zoom_y(self, factor):
        y0, y1 = self.ax.get_ylim()
        yc = 0.5 * (y0 + y1)
        half = (y1 - y0) * factor / 2.0
        self.ax.set_ylim(yc - half, yc + half)
        self.redraw_lines(); self.fig.canvas.draw_idle()

    def _shift_y(self, frac):
        y0, y1 = self.ax.get_ylim()
        dy = (y1 - y0) * frac
        self.ax.set_ylim(y0 + dy, y1 + dy)
        self.redraw_lines(); self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # terminal prompts
    # ------------------------------------------------------------------
    def _prompt_manual_redshift(self):
        try:
            val = input(f"[id {self.current_id}] Enter redshift: ")
            self.z = float(val)
            self.redraw_lines(); self.update_info_panel()
            self.fig.canvas.draw_idle()
        except ValueError:
            print("Invalid redshift value.")

    def _prompt_comment(self):
        self.pending_comment = input(f"[id {self.current_id}] Comment: ")
        print("Comment stored (saved with the next save action).")

    def _prompt_jump(self):
        val = input("Jump to id: ")
        try:
            target = int(val)
            if target in self.ids:
                self.idx = self.ids.index(target)
                self.load_current()
            else:
                print("id not found among discovered spectra.")
        except ValueError:
            print("Invalid id.")

    def _prompt_manual_coords(self):
        val = input("Enter RA, Dec in degrees (comma separated): ")
        try:
            ra_s, dec_s = val.split(',')
            self.ra, self.dec = float(ra_s), float(dec_s)
            print(f"Coordinates set: RA={self.ra}, Dec={self.dec}")
            self.update_info_panel()
        except Exception:
            print("Could not parse coordinates.")

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------
    def next_spectrum(self):
        if self.idx < len(self.ids) - 1:
            self.idx += 1
            self.load_current()
        else:
            print("Reached the end of the list.")

    def prev_spectrum(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_current()

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    def save_current(self, classification=None):
        oid = self.current_id
        prev_class = (self.catalog.loc[oid, 'classification']
                      if oid in self.catalog.index else '')
        row = {
            'id': oid,
            'z': self.z,
            'line': self.active_line if self.z is not None else '',
            'ra': self.ra,
            'dec': self.dec,
            'classification': classification if classification else prev_class,
            'comment': self.pending_comment,
            'simbad_match': self.last_simbad_match,
            'ned_match': self.last_ned_match,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        self.catalog.loc[oid] = row
        self._save_catalog()
        print(f"Saved id={oid}: z={self.z}, class={row['classification']}")
        self.update_info_panel()

    # ------------------------------------------------------------------
    # database queries
    # ------------------------------------------------------------------
    def query_simbad(self):
        if not (HAVE_SIMBAD and HAVE_ASTROPY):
            print("astroquery/astropy not available for SIMBAD query.")
            return
        if self.ra is None:
            print("No coordinates available - press 'k' to enter manually.")
            return
        try:
            coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
            res = Simbad.query_region(coord, radius=self.radius * u.arcsec)
            if res is None:
                print(f"SIMBAD: no match within {self.radius} arcsec.")
                self.last_simbad_match = 'none'
            else:
                name = res['main_id'][0]
                print(f"SIMBAD match: {name}")
                print(res)
                self.last_simbad_match = str(name)
        except Exception as e:
            print(f"SIMBAD query failed: {e}")
        self.update_info_panel()

    def query_ned(self):
        if not (HAVE_NED and HAVE_ASTROPY):
            print("astroquery/astropy not available for NED query.")
            return
        if self.ra is None:
            print("No coordinates available - press 'k' to enter manually.")
            return
        try:
            coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
            res = Ned.query_region(coord, radius=self.radius * u.arcsec)
            if res is None or len(res) == 0:
                print(f"NED: no match within {self.radius} arcsec.")
                self.last_ned_match = 'none'
            else:
                name = res['Object Name'][0]
                zz = res['Redshift'][0] if 'Redshift' in res.colnames else None
                print(f"NED match: {name}, z={zz}")
                print(res)
                self.last_ned_match = f"{name} z={zz}"
        except Exception as e:
            print(f"NED query failed: {e}")
        self.update_info_panel()

    def open_simbad_browser(self):
        if self.ra is None:
            print("No coordinates available - press 'k' to enter manually.")
            return
        url = (f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={self.ra}+{self.dec}"
               f"&Radius=2&Radius.unit=arcmin")
        webbrowser.open(url)

    def open_ned_browser(self):
        if self.ra is None:
            print("No coordinates available - press 'k' to enter manually.")
            return
        url = (f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search"
               f"&in_csys=Equatorial&in_equinox=J2000.0&lon={self.ra}d&lat={self.dec}d&radius=2")
        webbrowser.open(url)

    # ------------------------------------------------------------------
    # info panel / help
    # ------------------------------------------------------------------
    def update_info_panel(self):
        oid = self.current_id
        prior = self.catalog.loc[oid] if oid in self.catalog.index else None
        lines = [
            f"Object {self.idx + 1}/{len(self.ids)}  (id={oid})",
            f"Already saved: {'yes' if prior is not None else 'no'}",
        ]
        if prior is not None:
            lines.append(f"  prev class: {prior.get('classification','')}")
        lines += [
            "",
            f"RA, Dec: {self.ra}, {self.dec}" if self.ra is not None else "RA, Dec: unknown",
            "",
            f"Active line: {self.active_line}",
            f"z = {self.z:.4f}" if self.z is not None else "z = (not set)",
            "",
            f"SIMBAD: {self.last_simbad_match}",
            f"NED:    {self.last_ned_match}",
            "",
            "-- keys --",
            "click: set z (active line)",
            "1-5: Ha/Hb/MgII/Pab/Paa",
            "0: clear z   z: type z",
            "up/dn: zoom y  shift+up/dn: pan",
            "a: autoscale y",
            "n/right: next   b/left: prev",
            "j: jump to id",
            "s: save   x/g/u: star/gal/unk",
            "c: comment",
            "i/d: SIMBAD/NED query",
            "v/w: open SIMBAD/NED web",
            "k: manual RA/Dec",
            "h: help (console)",
        ]
        self.info_text.set_text("\n".join(lines))
        self.fig.tight_layout()

    def print_help(self):
        print(__doc__)


def main():
    p = argparse.ArgumentParser(description="SPHEREx spectrum inspector")
    p.add_argument('name', help="dataset name -> results_<name>/ directory")
    p.add_argument('--results_dir', default=None)
    p.add_argument('--outdir', default=None)
    p.add_argument('--start_id', type=int, default=None)
    p.add_argument('--radius', type=float, default=2.5,
                    help="SIMBAD/NED query radius in arcsec")
    args = p.parse_args()

    SpherexInspector(args.name, results_dir=args.results_dir,
                      outdir=args.outdir, start_id=args.start_id,
                      radius_arcsec=args.radius)


if __name__ == '__main__':
    main()
