"""
InsightX AI — Unit Tests for XAI Service (xai_service.py)

Tests the SHAP normalizer, LIME feature name parser, and explanation
generation logic.
"""
import numpy as np
import pytest

from app.services.xai_service import _normalize_shap_values, _extract_lime_feature_name


# ═══════════════════════════════════════════════════════════════════
#  _normalize_shap_values — unit tests
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeShapValues:
    """Tests for SHAP value normalization across different output formats."""

    def test_list_of_arrays_classification(self):
        """List of arrays (one per class) — pick predicted class."""
        # 2 classes, 3 features, 1 sample
        class0 = np.array([[0.1, 0.2, 0.3]])
        class1 = np.array([[0.4, 0.5, 0.6]])
        raw = [class0, class1]
        result = _normalize_shap_values(raw, is_classification=True, prediction=1, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.4, 0.5, 0.6])

    def test_list_of_arrays_regression(self):
        """List with single array for regression."""
        raw = [np.array([[0.1, 0.2, 0.3]])]
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_3d_array(self):
        """3D ndarray: (n_samples, n_features, n_classes)."""
        raw = np.array([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]])  # shape (1,3,2)
        result = _normalize_shap_values(raw, is_classification=True, prediction=1, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.4, 0.5, 0.6])

    def test_3d_array_class0(self):
        """3D ndarray, prediction=0."""
        raw = np.array([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]])
        result = _normalize_shap_values(raw, is_classification=True, prediction=0, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_2d_array(self):
        """2D ndarray: (n_samples, n_features) — regression case."""
        raw = np.array([[0.1, 0.2, 0.3]])
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_1d_array(self):
        """1D array — direct values."""
        raw = np.array([0.1, 0.2, 0.3])
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_truncation_to_n_features(self):
        """Output is truncated to n_features if SHAP returns more."""
        raw = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # 5 values
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        assert len(result) == 3

    def test_shap_explanation_object(self):
        """Handles shap.Explanation-like objects with a .values attribute."""
        class FakeExplanation:
            values = np.array([[0.7, 0.8, 0.9]])
        raw = FakeExplanation()
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        np.testing.assert_array_almost_equal(result, [0.7, 0.8, 0.9])

    def test_empty_fallback(self):
        """Returns zeros for unrecognized input shapes."""
        raw = np.float64(42.0)  # 0D scalar
        result = _normalize_shap_values(raw, is_classification=False, prediction=0, n_features=3)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════
#  _extract_lime_feature_name — unit tests
# ═══════════════════════════════════════════════════════════════════

class TestExtractLimeFeatureName:
    """Tests for parsing feature names from LIME condition strings."""

    def test_simple_greater_than(self):
        result = _extract_lime_feature_name("feature2 > 0.45", ["feature1", "feature2", "feature3"])
        assert result == "feature2"

    def test_simple_less_than(self):
        result = _extract_lime_feature_name("feature1 <= 0.30", ["feature1", "feature2"])
        assert result == "feature1"

    def test_range_condition(self):
        """LIME range: '0.10 < feature_name <= 0.50'"""
        result = _extract_lime_feature_name("0.10 < feature2 <= 0.50", ["feature1", "feature2"])
        assert result == "feature2"

    def test_one_hot_encoded_feature(self):
        """One-hot encoded feature like 'category_B'."""
        result = _extract_lime_feature_name("0.00 < category_B <= 1.00", ["category_B", "category_C", "feature1"])
        assert result == "category_B"

    def test_longest_match_first(self):
        """'feature_10' should match before 'feature_1'."""
        result = _extract_lime_feature_name("feature_10 > 5", ["feature_1", "feature_10", "feature_2"])
        assert result == "feature_10"

    def test_no_match_fallback(self):
        """When no known feature matches, fall back to parsing."""
        result = _extract_lime_feature_name("unknown_col > 0.5", ["feature1", "feature2"])
        assert result == "unknown_col "  or "unknown_col" in result

    def test_equality_condition(self):
        result = _extract_lime_feature_name("feature1 = 1.00", ["feature1", "feature2"])
        assert result == "feature1"
