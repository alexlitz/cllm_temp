"""
Verification utilities for weight compiler.

Provides bit-for-bit comparison between manually-set and compiled weights.
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WeightDiff:
    """Represents a difference in weight values."""
    layer_name: str
    param_name: str
    index: Tuple[int, ...]
    manual_value: float
    compiled_value: float
    diff: float

    def __repr__(self):
        return (
            f"{self.layer_name}.{self.param_name}[{self.index}]: "
            f"manual={self.manual_value:.6f}, compiled={self.compiled_value:.6f}, "
            f"diff={self.diff:.6f}"
        )


class Verifier:
    """Verifies compiled weights against manual implementation."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Maximum allowed difference for weight comparison.
        """
        self.tolerance = tolerance

    def compare_tensors(
        self,
        manual: torch.Tensor,
        compiled: torch.Tensor,
        name: str,
    ) -> List[WeightDiff]:
        """Compare two tensors and return list of differences.

        Args:
            manual: Tensor from manual weight setting
            compiled: Tensor from compiled weight setting
            name: Name for reporting

        Returns:
            List of WeightDiff objects for non-matching elements
        """
        diffs = []

        if manual.shape != compiled.shape:
            # Shape mismatch - report as single diff
            diffs.append(WeightDiff(
                layer_name=name,
                param_name="shape",
                index=(),
                manual_value=float(manual.numel()),
                compiled_value=float(compiled.numel()),
                diff=float('inf'),
            ))
            return diffs

        # Find element-wise differences
        diff_tensor = (manual - compiled).abs()
        mask = diff_tensor > self.tolerance

        if mask.any():
            indices = mask.nonzero(as_tuple=False)
            for idx in indices:
                idx_tuple = tuple(idx.tolist())
                diffs.append(WeightDiff(
                    layer_name=name,
                    param_name="",
                    index=idx_tuple,
                    manual_value=manual[idx_tuple].item(),
                    compiled_value=compiled[idx_tuple].item(),
                    diff=diff_tensor[idx_tuple].item(),
                ))

        return diffs

    def compare_layer(
        self,
        manual_layer,
        compiled_layer,
        layer_name: str,
    ) -> List[WeightDiff]:
        """Compare all parameters of a layer.

        Args:
            manual_layer: Layer module from manual model
            compiled_layer: Layer module from compiled model
            layer_name: Name for reporting

        Returns:
            List of all weight differences
        """
        all_diffs = []

        manual_params = dict(manual_layer.named_parameters())
        compiled_params = dict(compiled_layer.named_parameters())

        # Check for missing parameters
        for name in manual_params:
            if name not in compiled_params:
                all_diffs.append(WeightDiff(
                    layer_name=layer_name,
                    param_name=name,
                    index=(),
                    manual_value=1.0,
                    compiled_value=0.0,
                    diff=float('inf'),
                ))
                continue

            diffs = self.compare_tensors(
                manual_params[name].data,
                compiled_params[name].data,
                f"{layer_name}.{name}",
            )
            all_diffs.extend(diffs)

        return all_diffs

    def compare_attention(
        self,
        manual_attn,
        compiled_attn,
        layer_idx: int,
    ) -> List[WeightDiff]:
        """Compare attention layer weights.

        Args:
            manual_attn: Attention module from manual model
            compiled_attn: Attention module from compiled model
            layer_idx: Layer index for reporting

        Returns:
            List of weight differences
        """
        return self.compare_layer(
            manual_attn,
            compiled_attn,
            f"blocks[{layer_idx}].attn",
        )

    def compare_ffn(
        self,
        manual_ffn,
        compiled_ffn,
        layer_idx: int,
    ) -> List[WeightDiff]:
        """Compare FFN layer weights.

        Args:
            manual_ffn: FFN module from manual model
            compiled_ffn: FFN module from compiled model
            layer_idx: Layer index for reporting

        Returns:
            List of weight differences
        """
        return self.compare_layer(
            manual_ffn,
            compiled_ffn,
            f"blocks[{layer_idx}].ffn",
        )

    def compare_embedding(
        self,
        manual_embed,
        compiled_embed,
    ) -> List[WeightDiff]:
        """Compare embedding weights.

        Args:
            manual_embed: Embedding module from manual model
            compiled_embed: Embedding module from compiled model

        Returns:
            List of weight differences
        """
        return self.compare_layer(manual_embed, compiled_embed, "embed")

    def compare_output_head(
        self,
        manual_head,
        compiled_head,
    ) -> List[WeightDiff]:
        """Compare output head weights.

        Args:
            manual_head: head module from manual model
            compiled_head: head module from compiled model

        Returns:
            List of weight differences
        """
        return self.compare_layer(manual_head, compiled_head, "head")

    def compare_models(
        self,
        manual_model,
        compiled_model,
        verbose: bool = False,
    ) -> bool:
        """Compare all weights between manual and compiled models.

        Args:
            manual_model: Model with manually-set weights
            compiled_model: Model with compiled weights
            verbose: If True, print detailed differences

        Returns:
            True if weights match within tolerance
        """
        all_diffs = []

        # Compare embedding
        diffs = self.compare_embedding(
            manual_model.embed,
            compiled_model.embed,
        )
        all_diffs.extend(diffs)

        # Compare all transformer blocks
        num_blocks = len(manual_model.blocks)
        for i in range(num_blocks):
            # Attention
            diffs = self.compare_attention(
                manual_model.blocks[i].attn,
                compiled_model.blocks[i].attn,
                i,
            )
            all_diffs.extend(diffs)

            # FFN
            diffs = self.compare_ffn(
                manual_model.blocks[i].ffn,
                compiled_model.blocks[i].ffn,
                i,
            )
            all_diffs.extend(diffs)

        # Compare output head
        diffs = self.compare_output_head(
            manual_model.head,
            compiled_model.head,
        )
        all_diffs.extend(diffs)

        if verbose and all_diffs:
            print(f"Found {len(all_diffs)} weight differences:")
            for diff in all_diffs[:20]:  # Limit output
                print(f"  {diff}")
            if len(all_diffs) > 20:
                print(f"  ... and {len(all_diffs) - 20} more")

        return len(all_diffs) == 0

    def verify_layer_by_layer(
        self,
        manual_model,
        compiled_model,
    ) -> Dict[str, bool]:
        """Verify each layer individually and report status.

        Args:
            manual_model: Model with manually-set weights
            compiled_model: Model with compiled weights

        Returns:
            Dict mapping layer name to pass/fail status
        """
        results = {}

        # Embedding
        diffs = self.compare_embedding(
            manual_model.embed,
            compiled_model.embed,
        )
        results['embed'] = len(diffs) == 0

        # Transformer blocks
        num_blocks = len(manual_model.blocks)
        for i in range(num_blocks):
            # Attention
            diffs = self.compare_attention(
                manual_model.blocks[i].attn,
                compiled_model.blocks[i].attn,
                i,
            )
            results[f'L{i}.attn'] = len(diffs) == 0

            # FFN
            diffs = self.compare_ffn(
                manual_model.blocks[i].ffn,
                compiled_model.blocks[i].ffn,
                i,
            )
            results[f'L{i}.ffn'] = len(diffs) == 0

        # Output head
        diffs = self.compare_output_head(
            manual_model.head,
            compiled_model.head,
        )
        results['head'] = len(diffs) == 0

        return results

    def generate_diff_report(
        self,
        manual_model,
        compiled_model,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate detailed difference report.

        Args:
            manual_model: Model with manually-set weights
            compiled_model: Model with compiled weights
            output_path: Optional file path to write report

        Returns:
            Report string
        """
        lines = ["Weight Comparison Report", "=" * 50, ""]

        # Get layer-by-layer status
        status = self.verify_layer_by_layer(manual_model, compiled_model)

        passed = sum(1 for v in status.values() if v)
        total = len(status)

        lines.append(f"Summary: {passed}/{total} components match")
        lines.append("")

        # Detail each component
        for name, ok in status.items():
            status_str = "PASS" if ok else "FAIL"
            lines.append(f"  [{status_str}] {name}")

        # If failures, get details
        failed = [k for k, v in status.items() if not v]
        if failed:
            lines.append("")
            lines.append("Failure Details:")
            lines.append("-" * 40)

            # Collect all diffs
            all_diffs = []

            for name in failed:
                if name == 'embed':
                    diffs = self.compare_embedding(
                        manual_model.embed,
                        compiled_model.embed,
                    )
                elif name == 'head':
                    diffs = self.compare_output_head(
                        manual_model.head,
                        compiled_model.head,
                    )
                elif '.attn' in name:
                    idx = int(name.split('.')[0][1:])
                    diffs = self.compare_attention(
                        manual_model.blocks[idx].attn,
                        compiled_model.blocks[idx].attn,
                        idx,
                    )
                elif '.ffn' in name:
                    idx = int(name.split('.')[0][1:])
                    diffs = self.compare_ffn(
                        manual_model.blocks[idx].ffn,
                        compiled_model.blocks[idx].ffn,
                        idx,
                    )
                else:
                    continue

                all_diffs.extend(diffs)

            # Show top differences
            for diff in all_diffs[:50]:
                lines.append(f"  {diff}")

            if len(all_diffs) > 50:
                lines.append(f"  ... and {len(all_diffs) - 50} more differences")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def functional_test(
        self,
        model,
        bytecode: List[int],
        expected_ax: int,
        max_steps: int = 100,
    ) -> bool:
        """Run bytecode and verify AX output.

        Args:
            model: Neural VM model
            bytecode: List of bytecode instructions
            expected_ax: Expected AX register value
            max_steps: Maximum execution steps

        Returns:
            True if AX matches expected value
        """
        from ..run_vm import AutoregressiveVMRunner

        runner = AutoregressiveVMRunner(model=model)
        result = runner.run(bytecode, b'', [], '', max_steps=max_steps)

        actual_ax = result[1]
        return actual_ax == expected_ax
