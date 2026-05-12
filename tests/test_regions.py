import numpy as np
import pytest

from neuroacoustic_resonator import RegionMasks


def test_default_regions_have_expected_shape_and_do_not_overlap() -> None:
    regions = RegionMasks.from_size(8)

    overlap = (
        regions.input.astype(np.uint8)
        + regions.output.astype(np.uint8)
        + regions.assoc.astype(np.uint8)
    )

    assert regions.shape == (8, 8)
    assert np.all(overlap == 1)
    assert np.any(regions.input)
    assert np.any(regions.output)
    assert np.any(regions.assoc)


def test_default_regions_place_input_output_and_assoc_columns() -> None:
    regions = RegionMasks.from_size(5, edge_fraction=0.2)

    assert np.all(regions.input[:, 0])
    assert np.all(regions.output[:, -1])
    assert np.all(regions.assoc[:, 1:-1])
    assert not np.any(regions.input[:, 1:])
    assert not np.any(regions.output[:, :-1])


def test_region_mask_returns_copy() -> None:
    regions = RegionMasks.from_size(4)
    mask = regions.mask("input")

    mask[:, :] = False

    assert np.any(regions.input)


def test_region_labels_mark_each_region() -> None:
    regions = RegionMasks.from_size(3)
    labels = regions.labels()

    assert set(np.unique(labels)) == {1, 2, 3}
    assert np.all(labels[:, 0] == 1)
    assert np.all(labels[:, -1] == 2)
    assert np.all(labels[:, 1] == 3)


def test_regions_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        RegionMasks(
            input=np.ones((3, 3), dtype=np.bool_),
            output=np.ones((4, 4), dtype=np.bool_),
            assoc=np.ones((3, 3), dtype=np.bool_),
        )


def test_regions_reject_non_boolean_masks() -> None:
    with pytest.raises(ValueError, match="boolean"):
        RegionMasks(
            input=np.ones((3, 3), dtype=np.float64),
            output=np.zeros((3, 3), dtype=np.bool_),
            assoc=np.zeros((3, 3), dtype=np.bool_),
        )


def test_regions_reject_overlaps() -> None:
    input_mask = np.zeros((3, 3), dtype=np.bool_)
    output_mask = np.zeros((3, 3), dtype=np.bool_)
    assoc_mask = np.zeros((3, 3), dtype=np.bool_)
    input_mask[:, 0] = True
    output_mask[:, 0] = True
    assoc_mask[:, 1] = True

    with pytest.raises(ValueError, match="overlap"):
        RegionMasks(input=input_mask, output=output_mask, assoc=assoc_mask)


def test_default_regions_reject_too_small_size() -> None:
    with pytest.raises(ValueError, match="greater than 2"):
        RegionMasks.from_size(2)
