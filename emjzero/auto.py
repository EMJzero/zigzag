import os
import re
import shutil

def update_files(benchmark_convs, template_filename):
    # Read the template file contents
    with open(template_filename, 'r') as file:
        template_content = file.read()
    
    # Iterate through the dictionary entries
    for key, values in benchmark_convs.items():
        new_content = template_content
        string = ', '.join([str(values[k]) for k in ['M', 'P', 'Q', 'C', 'R', 'S']])
        new_content = re.sub(rf'loop_sizes\: \[1, 1, 1, 1, 1, 1, 1\]', f'loop_sizes: [1, {string}]', new_content)
        hs = values['Hstride']
        ws = values['Wstride']
        hd = values['Hdilation']
        wd = values['Wdilation']
        new_content = re.sub(rf'ix=1\*q\+1\*s, iy=1\*p\+1\*r', f'ix={ws}*q+{wd}*s, iy={hs}*p+{hd}*r', new_content)

        # Create new file with the dictionary key as name
        new_filename = f"{key}.yaml"
        with open(new_filename, 'w') as new_file:
            new_file.write(new_content)
        
        print(f"Created file: {new_filename}")

# Example dictionary
benchmark_convs = {
    # VGG16
    'I': dict(C =128, M = 256, P = 56, Q = 56, R = 3, S = 3, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    'II': dict(C =512, M = 512, P = 28, Q = 28, R = 3, S = 3, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    # ResNet18 and 50
    'III': dict(C =3, M = 64, P = 112, Q = 112, R = 7, S = 7, Hstride = 2, Wstride = 2, Hdilation = 1, Wdilation = 1),
    'IV': dict(C =64, M = 64, P = 56, Q = 56, R = 3, S = 3, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    'V': dict(C =128, M = 128, P = 28, Q = 28, R = 3, S = 3, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    'VI': dict(C =256, M = 256, P = 14, Q = 14, R = 3, S = 3, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    'VII': dict(C =256, M = 512, P = 7, Q = 7, R = 3, S = 3, Hstride = 2, Wstride = 2, Hdilation = 1, Wdilation = 1),
    'VIII': dict(C =64, M = 256, P = 56, Q = 56, R = 1, S = 1, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1), # depth-wise
    # MobileNetV3
    'IX': dict(C =3, M = 64, P = 112, Q = 112, R = 3, S = 3, Hstride = 2, Wstride = 2, Hdilation = 1, Wdilation = 1),
    'X': dict(C =72, M = 72, P = 28, Q = 28, R = 3, S = 3, Hstride = 2, Wstride = 2, Hdilation = 1, Wdilation = 1),
    'XI': dict(C =576, M = 576, P = 7, Q = 7, R = 5, S = 5, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1),
    'XII': dict(C =24, M = 88, P = 28, Q = 28, R = 1, S = 1, Hstride = 1, Wstride = 1, Hdilation = 1, Wdilation = 1), # depth-wise
    # Strides and Dilation
    'XIII': dict(C =16, M = 16, P = 224, Q = 224, R = 3, S = 3, Hstride = 3, Wstride = 3, Hdilation = 4, Wdilation = 4),
    'XIV': dict(C =128, M = 128, P = 112, Q = 112, R = 9, S = 9, Hstride = 4, Wstride = 4, Hdilation = 3, Wdilation = 3),
    'XV': dict(C =256, M = 256, P = 56, Q = 56, R = 3, S = 3, Hstride = 2, Wstride = 2, Hdilation = 3, Wdilation = 3)
}

template_filename = "template.txt"
update_files(benchmark_convs, template_filename)
