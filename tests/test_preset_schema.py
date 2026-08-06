"""Validate every shipped preset asset against the single canonical schema.

The whole point of the restructure is that all presets share *one* structure.
These tests enforce it two ways: a formal JSON Schema check (when `jsonschema`
is installed) and an always-on pure-Python structural check, so a malformed or
off-schema record is caught in CI rather than at import.
"""

from __future__ import annotations

import importlib.resources
import json

import pytest

#: The five preset assets loaded into `DATA_STYLES` (mirrors `colors._PRESET_ASSETS`).
ASSETS = [
    "ocean_presets.json",
    "terrain_presets.json",
    "ncl_presets.json",
    "weather_presets.json",
    "builtin_presets.json",
]

#: The complete set of keys a layer record may carry (the "one structure").
LAYER_KEYS = {
    "label", "colors", "colormap", "categories", "norm", "center", "vmin",
    "vmax", "levels", "bands", "extend", "alpha", "alpha_range", "units",
}
COLORMAPS = {"perceptual", "linear", "listed", "named"}


def _read(name: str) -> dict:
    """Read and parse a data-package JSON resource.

    Args:
        name: The resource filename inside `cleopatra.styling.data`.

    Returns:
        dict: The parsed JSON.
    """
    return json.loads(
        importlib.resources.files("cleopatra.styling.data").joinpath(name).read_text(encoding="utf-8")
    )


class TestPresetSchema:
    """Tests that every asset conforms to the canonical preset schema."""

    @pytest.mark.parametrize("asset", ASSETS)
    def test_validates_against_json_schema(self, asset):
        """Each asset validates against the shipped JSON Schema.

        Args:
            asset: The preset asset filename.

        Test scenario:
            The formal contract (`preset.schema.json`) accepts the asset; a
            structurally off-schema record would raise `ValidationError`.
        """
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(_read(asset), _read("preset.schema.json"))

    @pytest.mark.parametrize("asset", ASSETS)
    def test_uniform_record_shape(self, asset):
        """Every preset/layer in the asset shares the one canonical shape.

        Args:
            asset: The preset asset filename.

        Test scenario:
            Asset-level keys are provenance + presets; each preset is provenance
            + layers; each layer has `label` and is continuous (`colors` +
            `colormap`) XOR categorical (`categories`), with only allowed keys.
        """
        obj = _read(asset)
        assert set(obj) <= {"version", "source", "license", "presets"}, f"{asset}: stray asset key"
        assert obj.get("version") == 1, f"{asset}: missing/unknown version"
        for pname, preset in obj["presets"].items():
            assert set(preset) <= {"source", "license", "background", "layers"}, f"{asset}:{pname}: stray preset key"
            assert preset.get("layers"), f"{asset}:{pname}: no layers"
            for lname, layer in preset["layers"].items():
                where = f"{asset}:{pname}:{lname}"
                assert set(layer) <= LAYER_KEYS, f"{where}: stray layer key {set(layer) - LAYER_KEYS}"
                assert "label" in layer, f"{where}: missing label"
                is_cat = "categories" in layer
                is_cont = "colors" in layer and "colormap" in layer
                assert is_cat ^ is_cont, f"{where}: must be categorical XOR continuous"
                if is_cont:
                    assert layer["colormap"] in COLORMAPS, f"{where}: bad colormap {layer['colormap']!r}"
                    if isinstance(layer["colors"], str):
                        assert layer["colormap"] == "named", f"{where}: a colormap name needs colormap='named'"
                if is_cat:
                    for cat in layer["categories"]:
                        assert set(cat) == {"value", "color", "label"}, f"{where}: bad category {cat}"

    def test_schema_rejects_mixed_categorical_continuous(self):
        """The schema rejects a layer carrying both `categories` and `colors`.

        Test scenario:
            A self-contradictory layer (categorical AND continuous) must fail the
            `oneOf` contract -- the guarantee the whole restructure is built on.
        """
        jsonschema = pytest.importorskip("jsonschema")
        schema = _read("preset.schema.json")
        mixed = {"version": 1, "presets": {"x": {"layers": {"x": {
            "label": "mixed",
            "colors": ["#000000", "#ffffff"], "colormap": "listed",
            "categories": [{"value": 1, "color": "#000000", "label": "a"}],
        }}}}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(mixed, schema)

    def test_schema_couples_string_colors_to_named(self):
        """A string `colors` is only valid with `colormap="named"`.

        Test scenario:
            `{"colors": "viridis", "colormap": "listed"}` would drive the loader
            to build a colormap over the *characters* of the name, so the schema
            must reject it; the `named` form must still validate.
        """
        jsonschema = pytest.importorskip("jsonschema")
        schema = _read("preset.schema.json")

        def wrap(layer):
            return {"version": 1, "presets": {"x": {"layers": {"x": layer}}}}

        mangled = wrap({"label": "L", "colors": "viridis", "colormap": "listed"})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(mangled, schema)
        # the correct named form validates
        jsonschema.validate(wrap({"label": "L", "colors": "Spectral_r", "colormap": "named"}), schema)

    def test_schema_accepts_preset_background(self):
        """A preset may declare an optional string `background` (canvas colour).

        Test scenario:
            The flame presets pin a dark canvas via a preset-level `background`;
            the schema must accept a string there and reject a non-string.
        """
        jsonschema = pytest.importorskip("jsonschema")
        schema = _read("preset.schema.json")
        layer = {"label": "L", "colors": ["#000000", "#ffffff"], "colormap": "listed"}

        def wrap(preset):
            return {"version": 1, "presets": {"x": preset}}

        jsonschema.validate(wrap({"background": "#000000", "layers": {"x": layer}}), schema)
        bad = wrap({"background": 0, "layers": {"x": layer}})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_loader_carries_preset_background_onto_layers(self):
        """A preset-level `background` is copied onto each loaded layer config.

        Test scenario:
            `DATA_STYLES` is `{name: {layer: cfg}}` with no preset-level slot, so
            `_load_presets` must carry the canvas colour onto the layer(s) for the
            render path to find it.
        """
        from cleopatra.styling.colors import _load_presets

        flame = _load_presets("builtin_presets.json")["temperature_flame"]
        assert flame, "temperature_flame loaded with at least one layer"
        assert all(layer.get("background") == "#000000" for layer in flame.values())

    def test_total_precipitation_fades_at_low_end(self):
        """`total_precipitation` carries no constant alpha, so opacity tracks value.

        Test scenario:
            A precipitation total should read as rain over whatever is beneath it
            (a basemap) -- dry cells transparent, wet cells opaque. It therefore
            must not pin a constant `alpha`; with none, `alpha_scaled_image`
            defaults opacity to the colour norm.
        """
        from cleopatra.styling.colors import DATA_STYLES

        layer = DATA_STYLES["total_precipitation"]["total_precipitation"]
        assert "alpha" not in layer, "total_precipitation should fade at the low end (no constant alpha)"

    @pytest.mark.parametrize("asset", ASSETS)
    def test_loader_reads_asset(self, asset):
        """`_load_presets` builds at least one preset from each asset.

        Args:
            asset: The preset asset filename.

        Test scenario:
            The shipped asset loads (degrade-to-empty would signal a broken file).
        """
        from cleopatra.styling.colors import _load_presets

        loaded = _load_presets(asset)
        assert loaded, f"{asset}: loader produced no presets"
        for layers in loaded.values():
            for layer in layers.values():
                assert "label" in layer, f"{asset}: a loaded layer lacks a label"
