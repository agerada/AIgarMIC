Technical background
====================

In this section, we will review the algorithms that underpin ``AIgarMIC``. This section will not cover bacteriology or laboratory technique -- the reader is referred to the `validation manuscript <http://dx.doi.org/10.1128/spectrum.04209-23>`_ for detail on the experimental approach. Here, the focus is on the software algorithms that allow ``AIgarMIC`` to report minimum inhibitory concentrations (MICs).

Image splitting
---------------

The first key step that must be performed reliably is the splitting of an agar plate into smaller images representing a single inoculation site. Although fully automated image splitting algorithms were explored, such as the approach used by `cellprofiler <https://cellprofiler.org/>`_, the nature of agar dilution MICs generated some challenges. Firstly, such algorithms tend to rely on colony growth in at least three corner positions. With agar dilution, growth decreases in plates with higher antimicrobial concentration, therefore this approach cannot be relied upon in such plates.

Secondly, in real-world use, we often encounter stray colonies that would disturb algorithms that anchor to colony growth. For these reasons, we found the use of a black grid on the agar plate as the most practical and reliable method to direct an image splitting algorithm. We applied the grid (using transparent paper) during the photography step, allowing us to adjust the grid to compensate for stray colonies. The grid also worked well in plates with high antimicrobial concentrations, where growth can be sparse.

The algorithm first needs to map out the grid using `simple thresholding <https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html>`_. If the grid is added in software using a perfect black, this is easy, since a threshold value of ``0`` should work. If the grid is applied during photography, it may be an off black/grey color. To solve this issue, the algorithm searches for the lowest threshold value that returns the expected number of small images (split using `contours <https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html>`_: [#f1]_

.. code-block:: Delphi

    procedure find_threshold(image, max_threshold, n_rows, n_cols);
    for i := 1 to max_threshold do
        threshold_image := threshold(image, i);
        contours := find_contours(threshold_image);
        if length(contours) = n_rows * n_cols then
            return i;
        end
    end

Now that we have a threshold value that generates the expected number of contours, we can use the contours to split the image into small images. To make sure each image is in the correct position within the matrix that we generate, we iterate through the contours, sorted by their x-y co-ordinates: [#f2]_

.. code-block:: Delphi

    procedure split_by_grid(image, contours, n_rows, n_cols);
    image_matrix := array[1..n_rows, 1..n_cols] of image;
    contours := sort_left_to_right(contours);
    for i := 1 to n_rows do
        row_contours := sort_top_to_bottom(contours);
        for j := 1 to n_cols do
            image_matrix[i, j] := crop_image(image, row_contours[j]);
        end
    end

Finally, we end up with an matrix of images, each representing a single inoculation site.

MIC determination
-----------------

Assuming that we have an ``image_matrix`` for each agar plate in our experiment, we can combine these into a 3-dimensional matrix (in ``AIgarMIC`` this would correspond to a :class:`aigarmic.PlateSet` object). Assuming further that each ``image_matrix`` has been converted to a matrix of colony growth codes, the MIC determination algorithm can be applied.

We will assume that colony growth codes are one of ``(0, 1, 2)``, and that the threshold for uninhibited growth is ``2`` (i.e., values under ``2`` will be counted as inhibited growth). To calculate the MIC at position ``x, y``, we will search for the concentration at which growth starts to be inhibited. To do this, we will iterate in reverse until we find a concentration where growth is inhibited: [#f3]_

.. code-block:: Delphi

    plate_set := array[1..n_concentrations, 1..n_rows, 1..n_cols] of integer;
    plate_set := sort_by_descending_concentration(plate_set);
    inhibition_threshold := 2;
    procedure calculate_mic(plate_set, x, y);
        if plate_set[1, x, y] >= inhibition_threshold then
            return '>' + get_concentration(plate_set[1]);
        end
        mic := get_concentration(plate_set[1]);
        for i := 1 to length(plate_set) do
            if plate_set[i, x, y] >= inhibition_threshold then
                return mic;
            end
            mic := get_concentration(plate_set[i]);
        end
        return '<' + get_concentration(plate_set[n_rows]);
    end

Note that typically MIC results are manipulated as strings, since they can be above or  below the limit of detection. The algorithm firstly checks growth at the highest concentration, and if growth is not inhibited, returns a greater than this concentration result. Otherwise, it iterates through the concentrations until it finds uninhibited growth, and returns the previous (higher concentration) as the MIC. If no uninhibited growth is found (i.e., all of the plates have inhibited growth), it returns a less than the lowest concentration result.

Quality control
---------------

Quality assurance metrics are important to ensure that results generated by ``AIgarMIC`` are reliable and trustworthy. There are two quality assurance algorithms that we will outline, and are applied by default.

The first algorithm checks whether there is growth in the control plate (i.e., the plate with no antimicrobial). If there is no growth here, then the strain has failed QC, and returns ``F`` (fail): [#f4]_

.. code-block:: Delphi

    inhibition_threshold := 2;
    procedure check_control_growth(control_plate, x, y):
        if control_plate[x, y] < inhibition_threshold then
            return 'F';
        end
        return 'P';
    end

The second algorithm analyses the 3D matrix of colony growth codes (the ``plate_set``) and checks whether growth is consistent. A correct MIC experiment should have a concentration at which growth is inhibited (the MIC). All plates below this concentration should have good growth, and all plates above this concentration should have inhibited growth. If this is not the case (i.e., plates go from inhibited growth --> good growth --> inhibited growth), then the algorithm returns ``W`` (warning). Sometimes, this can be due to a technical error, such as a plate being accidentally missed during inoculation. The warning should prompt users to treat the result with some caution, perhaps double-checking it manually. To simplify the demonstration of this algorithm, we will assume that growth codes can be either ``0`` (inhibited growth) or ``1`` (uninhibited growth): [#f4]_

.. code-block:: Delphi

    plate_set := array[1..n_concentrations, 1..n_rows, 1..n_cols] of integer;
    plate_set := sort_by_descending_concentration(plate_set);
    procedure check_growth_consistency(plate_set, x, y):
        flips := 0;  { only one flip is allowed }
        previous_growth := plate_set[1, x, y];
        for i := 2 to length(plate_set) do
            if plate_set[i, x, y] <> previous_growth then
                flips := flips + 1;
            end
            previous_growth := plate_set[i, x, y];
        end
        if flips > 1 then
            return 'W';
        end
        return 'P';
    end

.. rubric:: Links to source code

.. [#f1] :func:`aigarmic.process_plate_image.find_threshold_value`
.. [#f2] :func:`aigarmic.process_plate_image.split_by_grid`
.. [#f3] :func:`aigarmic.plate.PlateSet.calculate_mic`
.. [#f4] :func:`aigarmic.plate.PlateSet.generate_qc`
