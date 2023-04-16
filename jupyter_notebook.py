"""Setup and tools for Jupyter notebooks"""
import random
import sys
import time
import warnings

import inspect
from pprint import pprint

import re
import numpy as np
import pandas as pd
import pandas.io.formats.style

pandas.options.display.max_columns = None
pandas.options.display.float_format = "{:,.2f}".format
from IPython.display import clear_output, display, HTML
from IPython.display import Markdown as md

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib import gridspec

# plt.style.use('seaborn')


# Inspired from https://www.overleaf.com/learn/latex/Font_sizes%2C_families%2C_and_styles#Reference_guide
class __fs:
    small = 12
    normal = 14
    large = 16


plt.fs = __fs()

plt.rcParams["font.size"] = str(plt.fs.normal)  # Default value for default values?
plt.rc("font", size=plt.fs.small)  # controls default text sizes
plt.rc("axes", titlesize=plt.fs.normal)  # fontsize of the axes title
plt.rc("axes", labelsize=plt.fs.normal)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=plt.fs.small)  # fontsize of the tick labels
plt.rc("ytick", labelsize=plt.fs.small)  # fontsize of the tick labels
plt.rc("legend", fontsize=plt.fs.small)  # legend fontsize
plt.rc("figure", titlesize=plt.fs.normal)  # fontsize of the figure title

## GENERAL HELPER FUNCTIONS ##

dollars = lambda amount: "{:,.2f}$".format(amount)

__tic = []  # Last in, first out (LIFO)


def tic():
    """Marks the beginning of a time interval"""
    global __tic
    __tic.append(time.perf_counter())


def toc(do_print=True):
    """Prints the time difference since the last tic (that was not toc'ed yet; LIFO)."""
    global __tic
    dt = time.perf_counter() - __tic.pop()
    if do_print:
        print("Elapsed time: %f seconds.\n" % dt)
    return dt


def pdb_on_warning():
    import pdb

    warnings.simplefilter(
        "error", [RuntimeWarning]
    )  # treat these warnings as exceptions
    try:
        pass  # Put you code here
    except:
        pdb.post_mortem(sys.exc_info()[-1])


def update_progress(progress, txt="Progress"):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = txt + ": [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text)


def split_duplicates(data):
    dup = data.index.duplicated(keep="first")
    if dup.sum() == 0:
        return data, pd.DataFrame([], index=data.index, columns=data.columns)
    return data[~dup], data[dup]


def multiple_replace(string, rep_dict):
    pattern = re.compile(
        "|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]),
        flags=re.DOTALL,
    )
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)


def plural(word, count=2):
    """https://www.geeksforgeeks.org/python-program-to-convert-singular-to-plural/"""
    if not (np.abs(count) > 1):
        return word

    # Check if word is ending with s,x,z or is
    # ending with ah, eh, ih, oh,uh,dh,gh,kh,ph,rh,th
    if re.search("[sxz]$", word) or re.search("[^aeioudgkprt]h$", word):
        return re.sub("$", "es", word)

    # Check if word is ending with ay,ey,iy,oy,uy
    if re.search("[aeiou]y$", word):
        # Make it plural by removing y from end adding ies to end
        return re.sub("y$", "ies", word)

    # Make the plural of word by adding s in end
    return word + "s"


def printdf(df, T=True):
    if isinstance(df, type(np.array([]))):
        df = pd.DataFrame(df)

    if isinstance(df, pd.Series):
        if T:
            df = df.to_frame().transpose()
        else:
            df = df.to_frame()

    if isinstance(df, pd.io.formats.style.Styler):
        display(df)
    else:
        display(HTML(df.to_html()))


def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + ".next()"

    toggle_text = "Toggle show/hide"  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ""  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += " next cell"
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = "code_toggle_{}".format(str(random.randint(1, 2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text,
    )

    return HTML(html)


class struct:
    """Matlab-inspired placeholder

    Using dictionaries would be more Pythonesque but makes the code a tad heavy at times. When
    efficiency is not a concern, we may use instances of this class.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        string = self.__class__.__name__ + "(\n"
        # breakpoint()
        is_public = lambda name: not (name.startswith("__") and name.endswith("__"))
        is_field = lambda name: hasattr(self, name) and not inspect.ismethod(
            getattr(self, name)
        )
        fields = self.__dict__
        for no, name in enumerate(fields):
            if is_public(name) and is_field(name):
                string += "    %s = %r" % (name, getattr(self, name))
            if no < len(fields) - 1:
                string += ","
            string += "\n"
        string += ")"
        return string

    def __repr__(self):
        return str(self).replace("\n", "").replace(" ", "")
