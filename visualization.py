from zigzag.utils import pickle_load
from zigzag.visualization.results.print_mapping import print_mapping
cmes = pickle_load("outputs/yaml-yaml-saved_list_of_cmes.pickle")
cme = cmes[0]
print_mapping(cme)