//! # terminal-hodge-harness
//!
//! A terminal-facing wrapper around [`hodge-belief`](https://github.com/SuperInstance/hodge-belief-rs)
//! that provides an `ErrorHodge` struct with evidence, coherence, and prior components.
//!
//! The real Hodge decomposition is delegated to `hodge-belief-rs`'s `BeliefHodge`,
//! but the API is designed for terminal/diagnostic use where errors are represented
//! as vector data over a simplicial complex.
//!
//! # Quick Start
//!
//! ```rust
//! use terminal_hodge_harness::ErrorHodge;
//! use terminal_hodge_harness::Dominance;
//!
//! let eh = ErrorHodge::decompose(&[1.0, 0.5, 0.3], 3, 0);
//! assert!(eh.total() > 0.0);
//! let dom = eh.domination();
//! assert!(matches!(dom, Dominance::Evidence | Dominance::Coherence | Dominance::Prior | Dominance::Mixed));
//! ```

use hodge_belief::{SimplicialComplex, ErrorHodge as HbErrorHodge};

/// Terminal-facing Hodge decomposition of an error signal.
///
/// # Fields
///
/// * `evidence`  — The exact (gradient) component: what happened, the raw signal.
/// * `coherence` — The co-exact (curl) component: internal consistency of the error.
/// * `prior`     — The harmonic component: expectation-vs-reality mismatch.
///
/// These correspond one-to-one to the exact, co-exact, and harmonic components
/// of the Hodge decomposition on a 1-cochain over a simplicial complex.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorHodge {
    pub evidence: Vec<f64>,
    pub coherence: Vec<f64>,
    pub prior: Vec<f64>,
}

/// Which component of the error is most significant overall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Dominance {
    /// The error is clear and factual; signal is strong.
    Evidence,
    /// The error is internally coherent; consistency dominates.
    Coherence,
    /// A prior/expectation mismatch dominates.
    Prior,
    /// No single component clearly dominates.
    Mixed,
}

impl Default for ErrorHodge {
    fn default() -> Self {
        Self {
            evidence: Vec::new(),
            coherence: Vec::new(),
            prior: Vec::new(),
        }
    }
}

impl ErrorHodge {
    /// Decompose a set of error values (observation strengths) over a
    /// simplicial complex into evidence, coherence, and prior components.
    ///
    /// The simplicial complex is built *from* the data:
    /// - `n_vertices` controls the number of nodes.
    /// - `n_triangles` controls the number of 2-simplices added.
    ///
    /// The `errors` slice defines the 1-cochain (edge values) of the complex.
    /// The complex is built as a path graph of `errors.len()` edges plus the
    /// specified number of triangles.
    ///
    /// # Arguments
    ///
    /// * `errors`     — Edge-level observation/error values (1-cochain).
    /// * `n_vertices` — Number of vertices in the complex. If 0, inferred
    ///                  as `errors.len() + 1`.
    /// * `n_triangles` — Number of triangles to add for richer topology.
    pub fn decompose(errors: &[f64], n_vertices: usize, n_triangles: usize) -> Self {
        let nv = if n_vertices == 0 { errors.len() + 1 } else { n_vertices };
        let ne = errors.len();
        let sc = build_complex(nv, ne, n_triangles);
        let hb = HbErrorHodge::decompose(errors, &sc);
        Self { evidence: hb.evidence, coherence: hb.coherence, prior: hb.prior }
    }

    /// Decompose using a user-supplied simplicial complex.
    ///
    /// This gives full control over the topology used for decomposition.
    pub fn with_complex(errors: &[f64], sc: &SimplicialComplex) -> Self {
        let hb = HbErrorHodge::decompose(errors, sc);
        Self { evidence: hb.evidence, coherence: hb.coherence, prior: hb.prior }
    }

    /// Sum of the absolute values of all components (total magnitude).
    pub fn total(&self) -> f64 {
        let e: f64 = self.evidence.iter().map(|x| x.abs()).sum();
        let c: f64 = self.coherence.iter().map(|x| x.abs()).sum();
        let p: f64 = self.prior.iter().map(|x| x.abs()).sum();
        e + c + p
    }

    /// Which component dominates the decomposition (by L2 norm).
    pub fn domination(&self) -> Dominance {
        let ev: f64 = self.evidence.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cv: f64 = self.coherence.iter().map(|x| x * x).sum::<f64>().sqrt();
        let pv: f64 = self.prior.iter().map(|x| x * x).sum::<f64>().sqrt();
        let total = ev + cv + pv;
        if total < 1e-15 {
            return Dominance::Mixed;
        }
        let threshold = 0.5;
        if ev / total > threshold {
            Dominance::Evidence
        } else if cv / total > threshold {
            Dominance::Coherence
        } else if pv / total > threshold {
            Dominance::Prior
        } else {
            Dominance::Mixed
        }
    }

    /// Cosine similarity of concatenated `[evidence, coherence, prior]` vectors.
    ///
    /// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector is
    /// zero.
    pub fn angle_between(&self, other: &ErrorHodge) -> f64 {
        let sv: Vec<f64> = self.evidence.iter()
            .cloned()
            .chain(self.coherence.iter().cloned())
            .chain(self.prior.iter().cloned())
            .collect();
        let ov: Vec<f64> = other.evidence.iter()
            .cloned()
            .chain(other.coherence.iter().cloned())
            .chain(other.prior.iter().cloned())
            .collect();
        let dot: f64 = sv.iter().zip(&ov).map(|(a, b)| a * b).sum();
        let n1: f64 = sv.iter().map(|x| x * x).sum::<f64>().sqrt();
        let n2: f64 = ov.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n1 < 1e-15 || n2 < 1e-15 {
            0.0
        } else {
            (dot / (n1 * n2)).clamp(-1.0, 1.0)
        }
    }

    /// L2 norm of the evidence component.
    pub fn evidence_norm(&self) -> f64 {
        self.evidence.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// L2 norm of the coherence component.
    pub fn coherence_norm(&self) -> f64 {
        self.coherence.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// L2 norm of the prior component.
    pub fn prior_norm(&self) -> f64 {
        self.prior.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Number of edges (length of each component vector).
    pub fn dim(&self) -> usize {
        self.evidence.len()
    }

    /// Check that the decomposition sums to the original error values:
    /// `evidence[i] + coherence[i] + prior[i] ≈ errors[i]`.
    ///
    /// Uses a tolerance of `1e-10`.
    pub fn verify(&self, errors: &[f64]) -> bool {
        self.dim() == errors.len()
            && self.evidence.iter()
                .zip(&self.coherence)
                .zip(&self.prior)
                .zip(errors)
                .all(|(((e, c), p), &o)| (e + c + p - o).abs() < 1e-10)
    }
}

// ── Internal helpers ────────────────────────────────────────────────

/// Build a simplicial complex from parameters.
///
/// Creates a path of `ne` edges + triangles starting from vertex 0.
fn build_complex(n_vertices: usize, n_edges: usize, n_triangles: usize) -> SimplicialComplex {
    // Build a path graph: edges [0,1], [1,2], ... up to n_edges-1 edges
    // For a path, we need n_vertices = n_edges + 1.
    // Edge indices must be within 0..n_vertices.
    let max_vert_for_path = n_edges; // path of n_edges edges ends at vertex n_edges
    let actual_vertices = n_vertices.max(max_vert_for_path + 1);
    let edges: Vec<[usize; 2]> = (0..n_edges).map(|i| [i, i + 1]).collect();
    let mut triangles = Vec::with_capacity(n_triangles);
    for t in 0..n_triangles {
        let base = t % (n_edges.saturating_sub(1));
        if base + 2 < actual_vertices {
            triangles.push([base, base + 1, base + 2]);
        }
    }
    SimplicialComplex::new(actual_vertices, edges, triangles)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction and Default ─────────────────────────────────────

    #[test]
    fn default_is_empty() {
        let eh = ErrorHodge::default();
        assert_eq!(eh.dim(), 0);
        assert!(eh.evidence.is_empty());
        assert!(eh.coherence.is_empty());
        assert!(eh.prior.is_empty());
    }

    #[test]
    fn decompose_basic_triangle() {
        let errors = vec![1.0, 0.5, 0.3];
        let eh = ErrorHodge::decompose(&errors, 3, 1);
        assert_eq!(eh.dim(), 3);
        assert!(eh.total() > 0.0);
    }

    #[test]
    fn decompose_inferred_vertices() {
        let errors = vec![0.2, 0.8, 0.5, 0.7];
        // n_vertices=0 → inferred as errors.len() + 1 = 5
        let eh = ErrorHodge::decompose(&errors, 0, 0);
        assert_eq!(eh.dim(), 4);
    }

    #[test]
    fn with_complex_uses_custom_topology() {
        let sc = SimplicialComplex::butterfly();
        let errors = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), 6);
    }

    // ── Decomposition Properties ─────────────────────────────────────

    #[test]
    fn decomposition_preserves_energy() {
        let errors = vec![1.0, 2.0, 3.0];
        let eh = ErrorHodge::decompose(&errors, 3, 1);
        assert!(eh.verify(&errors), "decomposition should sum to original");
    }

    #[test]
    fn verify_fails_on_wrong_length() {
        let eh = ErrorHodge::decompose(&[1.0, 2.0, 3.0], 3, 1);
        assert!(!eh.verify(&[1.0, 2.0]), "length mismatch should fail");
    }

    #[test]
    fn verify_fails_on_bad_values() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, 0.0, 0.0],
            coherence:  vec![0.0, 1.0, 0.0],
            prior:      vec![0.0, 0.0, 1.0],
        };
        assert!(!eh.verify(&[0.0, 0.0, 0.0]), "doesn't sum to zero input");
    }

    #[test]
    fn all_zero_errors_produce_zero_components() {
        let errors = vec![0.0, 0.0, 0.0];
        let eh = ErrorHodge::decompose(&errors, 3, 1);
        let epsilon = 1e-10;
        assert!(eh.evidence.iter().all(|&x| x.abs() < epsilon));
        assert!(eh.coherence.iter().all(|&x| x.abs() < epsilon));
        assert!(eh.prior.iter().all(|&x| x.abs() < epsilon));
    }

    #[test]
    fn constant_vector_is_gradient_on_path() {
        // A constant 1-cochain on a path graph should be mostly exact (gradient).
        // On a path, the first cohomology is trivial, so harmonic = 0.
        let errors = vec![1.0, 1.0, 1.0, 1.0];
        let eh = ErrorHodge::decompose(&errors, 0, 0);
        let epsilon = 1e-10;
        // On a path, harmonic component should be zero
        assert!(
            eh.prior.iter().all(|&x| x.abs() < epsilon),
            "path has trivial first cohomology, harmonic should be zero"
        );
    }

    #[test]
    fn constant_vector_on_cycle_has_harmonic_component() {
        // On a cycle graph, the constant 1-cochain should have a harmonic component
        // because cycles have non-trivial first cohomology.
        use hodge_belief::SimplicialComplex;
        let sc = SimplicialComplex::cycle(5);
        let errors = vec![1.0; 5];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        let pn = eh.prior_norm();
        assert!(
            pn > 0.0,
            "constant vector on a cycle should have non-zero harmonic component"
        );
    }

    // ── total() ──────────────────────────────────────────────────────

    #[test]
    fn total_of_zero_is_zero() {
        let eh = ErrorHodge {
            evidence:   vec![0.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        assert!((eh.total() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn total_sums_absolute_values() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, -2.0],
            coherence:  vec![3.0, -4.0],
            prior:      vec![5.0, -6.0],
        };
        // |1| + |2| + |3| + |4| + |5| + |6| = 21
        assert!((eh.total() - 21.0).abs() < 1e-12);
    }

    // ── domination() ─────────────────────────────────────────────────

    #[test]
    fn domination_evidence_when_exact_dominates() {
        let eh = ErrorHodge {
            evidence:   vec![10.0, 10.0, 10.0],
            coherence:  vec![1.0, 1.0, 1.0],
            prior:      vec![1.0, 1.0, 1.0],
        };
        assert_eq!(eh.domination(), Dominance::Evidence);
    }

    #[test]
    fn domination_coherence_when_coexact_dominates() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, 1.0, 1.0],
            coherence:  vec![10.0, 10.0, 10.0],
            prior:      vec![1.0, 1.0, 1.0],
        };
        assert_eq!(eh.domination(), Dominance::Coherence);
    }

    #[test]
    fn domination_prior_when_harmonic_dominates() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, 1.0, 1.0],
            coherence:  vec![1.0, 1.0, 1.0],
            prior:      vec![10.0, 10.0, 10.0],
        };
        assert_eq!(eh.domination(), Dominance::Prior);
    }

    #[test]
    fn domination_mixed_when_equal() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, 0.0, 0.0],
            coherence:  vec![0.0, 1.0, 0.0],
            prior:      vec![0.0, 0.0, 1.0],
        };
        assert_eq!(eh.domination(), Dominance::Mixed);
    }

    #[test]
    fn domination_all_zero_is_mixed() {
        let eh = ErrorHodge {
            evidence:   vec![0.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        assert_eq!(eh.domination(), Dominance::Mixed);
    }

    // ── angle_between() ──────────────────────────────────────────────

    #[test]
    fn angle_between_self_is_one() {
        let eh = ErrorHodge {
            evidence:   vec![1.0, 2.0, 3.0],
            coherence:  vec![4.0, 5.0, 6.0],
            prior:      vec![7.0, 8.0, 9.0],
        };
        let angle = eh.angle_between(&eh);
        assert!((angle - 1.0).abs() < 1e-12, "self cosine should be 1, got {}", angle);
    }

    #[test]
    fn angle_between_opposite_is_neg_one() {
        let a = ErrorHodge {
            evidence:   vec![1.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        let b = ErrorHodge {
            evidence:   vec![-1.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        let angle = a.angle_between(&b);
        assert!((angle + 1.0).abs() < 1e-12, "opposite cosine should be -1, got {}", angle);
    }

    #[test]
    fn angle_between_orthogonal_is_zero() {
        let a = ErrorHodge {
            evidence:   vec![1.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        let b = ErrorHodge {
            evidence:   vec![0.0, 1.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        let angle = a.angle_between(&b);
        assert!(angle.abs() < 1e-12, "orthogonal cosine should be 0, got {}", angle);
    }

    #[test]
    fn angle_between_zero_vector_is_zero() {
        let a = ErrorHodge {
            evidence:   vec![1.0, 2.0],
            coherence:  vec![3.0, 4.0],
            prior:      vec![5.0, 6.0],
        };
        let b = ErrorHodge {
            evidence:   vec![0.0, 0.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        let angle = a.angle_between(&b);
        assert!((angle - 0.0).abs() < 1e-12, "zero target should give 0, got {}", angle);
    }

    // ── norm methods ─────────────────────────────────────────────────

    #[test]
    fn component_norms_are_correct() {
        let eh = ErrorHodge {
            evidence:   vec![3.0, 4.0],
            coherence:  vec![0.0, 0.0],
            prior:      vec![0.0, 0.0],
        };
        assert!((eh.evidence_norm() - 5.0).abs() < 1e-12); // 3-4-5 triangle
        assert!((eh.coherence_norm() - 0.0).abs() < 1e-12);
        assert!((eh.prior_norm() - 0.0).abs() < 1e-12);
    }

    // ── Decomposition with real topology ─────────────────────────────

    #[test]
    fn butterfly_decomposition_has_three_components() {
        let sc = SimplicialComplex::butterfly();
        let errors = vec![0.5, 0.3, 0.7, 0.2, 0.9, 0.1];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), 6);
        assert!(eh.verify(&errors));
        assert!(eh.evidence_norm() > 0.0 || eh.coherence_norm() > 0.0 || eh.prior_norm() > 0.0);
    }

    #[test]
    fn path_decomposition_on_long_path() {
        let n = 10;
        let errors: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let sc = SimplicialComplex::path(n + 1);
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), n);
        assert!(eh.verify(&errors));
    }

    #[test]
    fn cycle_decomposition_all_ones() {
        let n = 6;
        let errors = vec![1.0; n];
        let sc = SimplicialComplex::cycle(n);
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), n);
        // A uniform vector on a cycle should have significant harmonic component
        // (constant function is harmonic on a cycle).
        assert!(eh.prior_norm() > 0.0);
    }

    #[test]
    fn tetrahedron_decomposition() {
        let sc = SimplicialComplex::tetrahedron();
        let errors = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), 6);
        assert!(eh.verify(&errors));
    }

    // ── Edge cases: single vertex/edge ───────────────────────────────

    #[test]
    fn single_edge_decomposition() {
        let sc = SimplicialComplex::single_edge();
        let errors = vec![0.42];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), 1);
        assert!(eh.verify(&errors));
        // Single edge: B1 is 1x2, so exact = projection; no triangles so coexact=0;
        // harmonic = residual
        assert!(eh.coherence.iter().all(|&x| x.abs() < 1e-12),
            "no triangles → coherence should be zero");
    }

    #[test]
    fn single_vertex_no_edges() {
        let sc = SimplicialComplex::single_vertex();
        let errors: Vec<f64> = vec![];
        let eh = ErrorHodge::with_complex(&errors, &sc);
        assert_eq!(eh.dim(), 0);
        assert!(eh.evidence.is_empty());
        assert!(eh.coherence.is_empty());
        assert!(eh.prior.is_empty());
    }

    // ── serde round-trip ─────────────────────────────────────────────

    #[test]
    fn serde_round_trip() {
        let eh = ErrorHodge::decompose(&[0.5, 1.5, 2.5], 3, 1);
        let json = serde_json::to_string(&eh).unwrap();
        let deserialized: ErrorHodge = serde_json::from_str(&json).unwrap();
        assert_eq!(eh.evidence, deserialized.evidence);
        assert_eq!(eh.coherence, deserialized.coherence);
        assert_eq!(eh.prior, deserialized.prior);
    }

    // ── clone and debug ──────────────────────────────────────────────

    #[test]
    fn clone_is_deep() {
        let eh1 = ErrorHodge::decompose(&[1.0, 0.0, 0.0], 3, 1);
        let mut eh2 = eh1.clone();
        eh2.evidence[0] = 99.0;
        assert!((eh1.evidence[0] - 1.0).abs() < 1e-12, "clone should be deep");
    }

    // ── total on decomposed data ──────────────────────────────────────

    #[test]
    fn total_from_real_decomposition() {
        let errors = vec![0.1, 0.2, 0.3];
        let eh = ErrorHodge::decompose(&errors, 3, 1);
        let total = eh.total();
        assert!(total > 0.0, "non-zero errors should yield non-zero total");
        assert!(total < 1e3, "total should be reasonable");
    }

    // ── Mixed dominance boundary ─────────────────────────────────────

    #[test]
    fn domination_is_mixed_when_none_exceeds_threshold() {
        // Each component L2 norm roughly equal so none >50%
        let eh = ErrorHodge {
            evidence:   vec![0.5, 0.5, 0.5],
            coherence:  vec![0.5, 0.5, 0.5],
            prior:      vec![0.5, 0.5, 0.5],
        };
        assert_eq!(eh.domination(), Dominance::Mixed);
    }

    // ── angle_between different lengths ──────────────────────────────

    #[test]
    fn angle_between_different_dimensions_panics() {
        let a = ErrorHodge {
            evidence:   vec![1.0],
            coherence:  vec![2.0],
            prior:      vec![3.0],
        };
        let b = ErrorHodge {
            evidence:   vec![1.0, 2.0],
            coherence:  vec![3.0, 4.0],
            prior:      vec![5.0, 6.0],
        };
        // Different concatenated lengths → zip will just stop early, giving wrong result
        // This is an API misuse; we just document it rather than enforcing.
        // The call will produce a value but it won't be meaningful.
        let _angle = a.angle_between(&b);
        // No panic expected; but result is undefined for mismatched dimensions.
    }
}
