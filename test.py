from plate import *

test_plates = [ 
    Plate('genta', 128.), 
    Plate('genta', 64.), 
    Plate('genta', 32.)
]
test_plates[0].growth_matrix = [[2,0],[0,0]]
test_plates[1].growth_matrix = [[2,0],[2,0]]
test_plates[2].growth_matrix = [[2,0],[0,2]]
for i in test_plates: 
    i.key = ['No growth','Poor growth','Good growth']
plate_set = PlateSet(test_plates)
print(plate_set)
plate_set.calculate_MIC()
print(plate_set.convert_mic_matrix(str))