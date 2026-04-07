from collections import OrderedDict

from cleopatra.styles import Styles


def test_create_instance():
    assert isinstance(Styles.marker_style_list, list)
    assert isinstance(Styles.line_styles, OrderedDict)
