MIC calculation
===============

In its most minimal form, ``AIgarMIC`` can be used as a simple agar dilution MI calculator. Here, it is up to users to provide growth matrices for each agar dilution plate. Such users can then use :class:`aigarmic.Plate` and :class:`aigarmic.PlateSet` to calculate MICs:

>>> from aigarmic import Plate, PlateSet
>>> antibiotic = "amikacin"
>>> plate1 = Plate(drug=antibiotic,
...                 concentration=0,
...                 growth_code_matrix=[[2, 2, 2],
...                                [2, 2, 0],
...                                [0, 2, 2]])
>>> plate2 = Plate(drug=antibiotic,
...                 concentration=0.125,
...                 growth_code_matrix=[[0, 1, 2],
...                                [2, 2, 0],
...                                [0, 2, 2]])
>>> plate3 = Plate(drug=antibiotic,
...                 concentration=0.25,
...                 growth_code_matrix=[[0, 1, 2],
...                                [2, 2, 0],
...                                [0, 2, 1]])
>>> plate4 = Plate(drug=antibiotic,
...                 concentration=0.5,
...                 growth_code_matrix=[[0, 0, 2],
...                                [2, 0, 0],
...                                [0, 2, 0]])
>>> plate_set = PlateSet(plates_list=[plate1, plate2, plate3, plate4])
>>> plate_set.calculate_mic(no_growth_key_items=tuple([0, 1]))
array([[0.125, 0.125, 1.   ],
       [1.   , 0.5  , 0.125],
       [0.125, 1.   , 0.25 ]])

As we can see, :func:`aigarmic.PlateSet.calculate_mic()` returns a numpy array with the MICs for each strain in the plate set. This is a 'numerical' MIC -- strains with growth in the plate with highest antibiotic concentration are reported as double the highest concentration. To convert to a more classical string MIC format, with appropriate censoring (e.g., '<0.125', '>0.5'), we can use :func:`aigarmic.PlateSet.convert_mic_matrix()`:

>>> plate_set.convert_mic_matrix(mic_format='string').tolist()
[['<0.125', '<0.125', '>0.5'], ['>0.5', '0.5', '<0.125'], ['<0.125', '>0.5', '0.25']]
