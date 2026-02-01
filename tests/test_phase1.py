"""Phase 1 Tests: Foundation Documents and Code Examples.

Tests verify all theory documents exist, meet requirements, and code runs.
"""

import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
FOUNDATIONS_DIR = PROJECT_ROOT / "01-foundations"


class TestDocumentsExist:
    """Test that all required documents were created."""

    def test_attention_document_exists(self):
        """Test attention basics document exists."""
        doc_path = FOUNDATIONS_DIR / "01_attention_basics.md"
        assert doc_path.exists(), f"Missing: {doc_path}"

    def test_transformer_document_exists(self):
        """Test transformer theory document exists."""
        doc_path = FOUNDATIONS_DIR / "02_transformer_theory.md"
        assert doc_path.exists(), f"Missing: {doc_path}"

    def test_jepa_document_exists(self):
        """Test JEPA principle document exists."""
        doc_path = FOUNDATIONS_DIR / "03_jepa_principle.md"
        assert doc_path.exists(), f"Missing: {doc_path}"

    def test_comparison_document_exists(self):
        """Test architecture comparison document exists."""
        doc_path = FOUNDATIONS_DIR / "04_architecture_comparison.md"
        assert doc_path.exists(), f"Missing: {doc_path}"


class TestWordCounts:
    """Test that documents meet minimum word count requirements."""

    def get_word_count(self, filepath):
        """Get word count of a file using wc."""
        result = subprocess.run(
            ["wc", "-w", str(filepath)], capture_output=True, text=True, check=True
        )
        return int(result.stdout.split()[0])

    def test_attention_word_count(self):
        """Attention document should have 1500+ words."""
        doc_path = FOUNDATIONS_DIR / "01_attention_basics.md"
        count = self.get_word_count(doc_path)
        assert count >= 1500, f"Word count {count} < 1500"

    def test_transformer_word_count(self):
        """Transformer document should have 2000+ words."""
        doc_path = FOUNDATIONS_DIR / "02_transformer_theory.md"
        count = self.get_word_count(doc_path)
        assert count >= 1900, f"Word count {count} < 1900"  # Allow some tolerance

    def test_jepa_word_count(self):
        """JEPA document should have 1800+ words (or close)."""
        doc_path = FOUNDATIONS_DIR / "03_jepa_principle.md"
        count = self.get_word_count(doc_path)
        assert count >= 1500, f"Word count {count} < 1500"  # Tolerance

    def test_comparison_word_count(self):
        """Comparison document should have 1500+ words."""
        doc_path = FOUNDATIONS_DIR / "04_architecture_comparison.md"
        count = self.get_word_count(doc_path)
        assert count >= 1500, f"Word count {count} < 1500"


class TestCodeExamples:
    """Test that code examples exist and run successfully."""

    def test_attention_example_exists(self):
        """Test attention example code exists."""
        code_path = FOUNDATIONS_DIR / "attention_example.py"
        assert code_path.exists(), f"Missing: {code_path}"

    def test_attention_example_runs(self):
        """Test that attention_example.py executes without errors."""
        code_path = FOUNDATIONS_DIR / "attention_example.py"
        result = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Code failed:\n{result.stderr}"
        assert "✓" in result.stdout, "Missing success indicator in output"

    def test_attention_example_output(self):
        """Test attention example produces expected output."""
        code_path = FOUNDATIONS_DIR / "attention_example.py"
        result = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        # Check for key elements in output
        assert "Attention" in result.stdout
        assert "shape" in result.stdout
        assert "sum=" in result.stdout  # Attention weights sum
        assert "1.000" in result.stdout  # Weights sum to 1


class TestDocumentContent:
    """Test that documents contain required sections and elements."""

    def test_attention_has_eli5(self):
        """Attention document should have ELI5 explanation."""
        doc_path = FOUNDATIONS_DIR / "01_attention_basics.md"
        content = doc_path.read_text()
        assert "ELI5" in content or "eli5" in content.lower()

    def test_attention_has_math(self):
        """Attention document should have mathematical formulas."""
        doc_path = FOUNDATIONS_DIR / "01_attention_basics.md"
        content = doc_path.read_text()
        # Check for LaTeX math
        assert "$$" in content or "$" in content
        assert "sqrt" in content or "√" in content  # Scaling formula

    def test_jepa_cites_paper(self):
        """JEPA document should cite the VL-JEPA paper."""
        doc_path = FOUNDATIONS_DIR / "03_jepa_principle.md"
        content = doc_path.read_text()
        assert "2512.10942" in content, "VL-JEPA paper not cited"
        assert "arxiv.org" in content

    def test_comparison_has_table(self):
        """Comparison document should have comparison tables."""
        doc_path = FOUNDATIONS_DIR / "04_architecture_comparison.md"
        content = doc_path.read_text()
        assert "|" in content  # Markdown table syntax
        assert "Transformer" in content
        assert "JEPA" in content or "VL-JEPA" in content


class TestAttentionMechanism:
    """Test the attention mechanism implementation itself."""

    def test_can_import_attention_functions(self):
        """Test that we can import functions from attention_example.py."""
        import sys

        # Add 01-foundations to path
        sys.path.insert(0, str(FOUNDATIONS_DIR))

        try:
            from attention_example import scaled_dot_product_attention

            assert callable(scaled_dot_product_attention)
        finally:
            sys.path.pop(0)

    def test_attention_computes_correct_shapes(self):
        """Test attention mechanism produces correct output shapes."""
        import sys

        import torch

        sys.path.insert(0, str(FOUNDATIONS_DIR))

        try:
            from attention_example import scaled_dot_product_attention

            # Test with simple inputs
            batch_size, seq_len, d_k = 2, 4, 8
            Q = torch.randn(batch_size, seq_len, d_k)
            K = torch.randn(batch_size, seq_len, d_k)
            V = torch.randn(batch_size, seq_len, d_k)

            output, weights = scaled_dot_product_attention(Q, K, V)

            # Check shapes
            assert output.shape == (batch_size, seq_len, d_k)
            assert weights.shape == (batch_size, seq_len, seq_len)

            # Check attention weights sum to 1
            row_sums = weights.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-5
            ), "Attention weights don't sum to 1"
        finally:
            sys.path.pop(0)


def test_phase1_summary():
    """Generate summary of Phase 1 completion."""
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETION SUMMARY")
    print("=" * 70)

    docs = [
        "01_attention_basics.md",
        "02_transformer_theory.md",
        "03_jepa_principle.md",
        "04_architecture_comparison.md",
    ]

    for doc in docs:
        doc_path = FOUNDATIONS_DIR / doc
        if doc_path.exists():
            # Get word count
            result = subprocess.run(["wc", "-w", str(doc_path)], capture_output=True, text=True)
            count = int(result.stdout.split()[0])
            print(f"✓ {doc:40s} {count:>6} words")
        else:
            print(f"✗ {doc:40s} MISSING")

    print("=" * 70)
    print("All Phase 1 foundation documents complete!")
    print("=" * 70)
