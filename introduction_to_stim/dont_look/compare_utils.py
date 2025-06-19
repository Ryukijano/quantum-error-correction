from io import BytesIO
from cairosvg import svg2png
from jupyter_compare_view import compare
import matplotlib.image as mpimg
import numpy as np
from .correct_rep_code import create_rep_code_stim_string
import stim

def compare_svg_diagram(d1, d2):
    with BytesIO(svg2png(bytestring=d1._repr_svg_())) as fp:
        img = mpimg.imread(fp, format='png')

    with BytesIO(svg2png(bytestring=d2._repr_svg_())) as fp:
        img2 = mpimg.imread(fp, format='png')

    print(f"Your circuit's diagram (left) {'IS' if img.shape == img2.shape and np.all(img == img2) else 'is NOT'} \
identical to the reference one (right).")
    return compare(img,
                 img2,
                 start_mode='horizontal',
                 height=285,
                 display_format='png',
                 add_controls=False,
                 )

def compare_plt_fig(fig1, filename):
    b1 = BytesIO()
    fig1.savefig(b1, format='png')
    b1.seek(0)
    img1 = b1.getvalue()
    b1.close()
    
    with open(filename, 'rb') as fp:
        img2 = fp.read()
    
    return compare(img1,
                 img2,
                 start_mode='horizontal',
                 #height=285,
                 display_format='png',
                 add_controls=False,
                 )

def compare_circuit_timelines(circuit1, circuit2):
    d1, d2 = circuit1.diagram('timeline-svg'), circuit2.diagram('timeline-svg')
    return compare_svg_diagram(d1, d2)

def compare_part1(circuit):
    return compare_circuit_timelines(circuit, stim.Circuit.from_file('dont_look/correct_part1.stim'))
    
def compare_part2(circuit):
    return compare_circuit_timelines(circuit, stim.Circuit.from_file('dont_look/correct_part2.stim'))

def compare_part3(circuit):
    return compare_circuit_timelines(circuit, stim.Circuit.from_file('dont_look/correct_part3.stim'))

def compare_repcode(circuit, distance, rounds, p):
    return compare_circuit_timelines(circuit, stim.Circuit(create_rep_code_stim_string(distance, rounds, p)))

def compare_repcode_plot1(fig):
    return compare_plt_fig(fig, 'dont_look/repcode_plot1.png')
    
def compare_repcode_plot2(fig):
    return compare_plt_fig(fig, 'dont_look/repcode_plot2.png')
    
def compare_repcode_plot3(fig):
    return compare_plt_fig(fig, 'dont_look/repcode_plot3.png')