"""
Test loading all compiled weights into VM.
"""

from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM

def test_load_all_weights():
    """Test loading all 32 compiled opcode weights."""
    
    print("Creating AutoregressiveVM with d_model=1280...")
    vm = AutoregressiveVM(
        d_model=1280,
        n_layers=16,
        n_heads=8,
        ffn_hidden=4096,
    )
    print()
    
    loader = CompiledWeightLoader()
    stats = loader.load_all_weights(vm, verbose=True)
    
    print()
    print("Testing layer access...")
    layer_map = loader.get_layer_mapping()
    for name, layer_idx in layer_map.items():
        layer = vm.blocks[layer_idx]
        ffn_params = sum(p.numel() for p in layer.ffn.parameters())
        print(f"  {name:20s} (Layer {layer_idx}): {ffn_params:,} total params")
    
    return stats

if __name__ == "__main__":
    stats = test_load_all_weights()
    
    total_loaded = stats['single_op_loaded'] + stats['multi_op_loaded'] + stats['multi_layer_loaded']
    success = (total_loaded == 32 and len(stats['failed']) == 0)
    exit(0 if success else 1)
