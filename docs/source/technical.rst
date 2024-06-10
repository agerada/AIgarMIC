Technical background
====================

In this section, we will review the algorithms that underpin ``AIgarMIC``. This section will not cover bacteriology or laboratory technique -- the reader is referred to the `validation manuscript <http://dx.doi.org/10.1128/spectrum.04209-23>`_ for detail on the experimental approach. Here, the focus is on the software algorithms that allow ``AIgarMIC`` to report minimum inhibitory concentrations (MICs).

Image splitting
---------------

The first key step that must be performed reliably is the splitting of an agar plate into smaller images representing a single inoculation site. Although fully automated image splitting algorithms were explored, such as the approach used by `cellprofiler <https://cellprofiler.org/>`_, the nature of agar dilution MICs generated some challenges. Firstly, such algorithms tend to rely on colony growth in at least three corner positions. With agar dilution, growth decreases in plates with higher antimicrobial concentration, therefore this approach cannot be relied upon in such plates.

Secondly, in real-world use, we often encounter stray colonies that would disturb algorithms that anchor to colony growth. For these reasons, we found the use of a black grid on the agar plate as the most practical and reliable method to direct an image splitting algorithm. We applied the grid (using transparent paper) during the photography step, allowing us to adjust the grid to compensate for stray colonies. The grid also worked well in plates with high antimicrobial concentrations, where growth can be sparse.

The algorithm first needs to map out the grid using `simple thresholding <https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html>`_. If the grid is added in software using a perfect black, this is easy, since a threshold value of ``0`` should work. If the grid is applied during photography, it may be an off black/grey color. To solve this issue, the algorithm searches for the lowest threshold value that returns the expected number of small images (split using `contours <https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html>`_::

    procedure find_threshold(image, max_threshold, n_rows, n_cols);
    for i := 1 to max_threshold do
        threshold_image := threshold(image, i);
        contours := find_contours(threshold_image);
        if length(contours) = n_rows * n_cols then
            return i;
        end
    end

Now that we have a threshold value that generates the expected number of contours, we can use the contours to split the image into small images. To make sure each image is in the correct position within the matrix that we generate, we iterate through the contours, sorted by their x-y co-ordinates::

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

MIC determination algorithm
---------------------------

Assuming that we have an ``image_matrix`` for each agar plate in our experiment, we can combine these into a 3-dimensional matrix (in ``AIgarMIC`` this would correspond to a :class:`aigarmic.PlateSet` object). Assuming further that each ``image_matrix`` has been converted to a matrix of colony growth codes, the MIC determination algorithm can be applied.

We will assume that colony growth codes are one of ``(0, 1, 2)``, and that the threshold for uninhibited growth is ``2`` (i.e., values under ``2`` will be counted as inhibited growth). To calculate the MIC at position ``x, y``, we will search for the concentration at which growth starts to be inhibited. To do this, we will iterate in reverse until we find a concentration where growth is inhibited::

    plate_set := array[1..n_concentrations, 1..n_rows, 1..n_cols] of integer;
    plate_set := sort_by_descending_concentration(plate_set);
    inhibition_threshold := 2;
    procedure calculate_mic(plate_set, x, y);
        if plate_set[1, x, y] >= inhibition_threshold then
            return "MIC > " + get_concentration(plate_set[1]);
        end
        mic := get_concentration(plate_set[1]);
        for i := 1 to length(plate_set) do
            if plate_set[i, x, y] >= inhibition_threshold then
                return mic;
            end
            mic := get_concentration(plate_set[i]);
        end
        return "MIC < " + get_concentration(plate_set[n_rows]);
    end

Note that typically MIC results are manipulated as strings, since they can be above or  below the limit of detection. The algorithm firstly checks growth at the highest concentration, and if growth is not inhibited, returns a greater than this concentration result. Otherwise, it iterates through the concentrations until it finds uninhibited growth, and returns the previous (higher concentration) as the MIC. If no uninhibited growth is found (i.e., all of the plates have inhibited growth), it returns a less than the lowest concentration result.

