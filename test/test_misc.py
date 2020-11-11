import fteikpy


def test_num_threads():
    fteikpy.set_num_threads(1)

    assert fteikpy.get_num_threads() == 1
