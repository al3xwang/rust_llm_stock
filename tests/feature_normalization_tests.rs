use rust_llm_stock::feature_normalization::{FEATURE_SIZE, normalize_features};

#[test]
fn industry_features_clamp_and_scale() {
    // Prepare a feature vector with out-of-range industry values
    let mut features = [0.0f32; FEATURE_SIZE];

    // Set exaggerated industry values to ensure clamping kicks in
    features[91] = 0.15; // > +0.1 -> should clamp to +0.1 -> normalized = +1.0
    features[92] = -0.25; // < -0.2 -> should clamp to -0.2 -> normalized = -1.0
    features[93] = 0.3; // > +0.2 -> should clamp to +0.2 -> normalized = +1.0

    // reference_close_pct not used for these indices but pass a reasonable value
    let normalized = normalize_features(features, 0.02);

    assert!((normalized[91] - 1.0).abs() < 1e-6, "index 91 expected 1.0, got {}", normalized[91]);
    assert!((normalized[92] + 1.0).abs() < 1e-6, "index 92 expected -1.0, got {}", normalized[92]);
    assert!((normalized[93] - 1.0).abs() < 1e-6, "index 93 expected 1.0, got {}", normalized[93]);
}
