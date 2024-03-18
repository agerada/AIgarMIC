test_plates = [
    Plate('genta', 128.), 
    Plate('genta', 64.), 
    Plate('genta', 32.), 
    Plate('genta', 16.),
    Plate('genta', 0.),
]

test_plates[0].growth_code_matrix = [
    [0,0],
    [0,0]]
test_plates[1].growth_code_matrix = [
    [1,2],
    [2,0]]
test_plates[2].growth_code_matrix = [
    [2,2],
    [0,0]]
test_plates[3].growth_code_matrix = [
    [2,2],
    [2,0]]

test_plates[4].growth_code_matrix = [
    [2,2],
    [2,2]]

for p in test_plates: 
    p.growth_matrix = [[0,0],[0,0]]
    p.key = ['No growth','Poor growth','Good growth']
    for i,row in enumerate(p.growth_code_matrix): 
        for j,item in enumerate(row): 
            p.growth_matrix[i][j] = p.key[p.growth_code_matrix[i][j]]

plate_set = PlateSet(test_plates)
plate_set.calculate_MIC()
print(plate_set.generate_QC())

csv_data = plate_set.get_csv_data()
exit()
with open('test_dict.csv', 'a') as csvfile: 
    writer = csv.DictWriter(csvfile, csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)

